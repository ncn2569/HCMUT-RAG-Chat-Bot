import json
import os
import torch
import numpy as np
import threading
from google.genai import types
from rag.retrieval.rerank import model, tokenizer
from rag.embedding.embed import load_embedder

CACHE_FILE = "data/semantic_cache.jsonl"
embedder = load_embedder()


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)


class SemanticCache:
    def __init__(self, threshold=0.85, max_size=20):
        self.threshold = threshold
        self.max_size = max_size
        self.cache = []  # Danh sách chứa dict: {"query": str, "answer": str, "embedding": list}
        self.lock = threading.Lock()
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.cache.append(json.loads(line))
            # Cắt bớt nếu số lượng nạp vào vượt quá max_size
            if len(self.cache) > self.max_size:
                self.cache = self.cache[-self.max_size :]

    def _save_cache_to_disk(self):
        # Ghi đè lại toàn bộ file với danh sách đã bị giới hạn 20 phần tử
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            for entry in self.cache:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def check(self, current_query):
        with self.lock:
            cache_copy = list(self.cache)

        if not cache_copy:
            return None

        # 1. Exact match (Nhanh tuyệt đối)
        for item in cache_copy:
            if item["query"].lower().strip() == current_query.lower().strip():
                # print("Đã tìm thấy trong Cache (Exact Match)!")
                return item["answer"]

        # 2. Vector Search (Lấy Top 5)
        # print(" Đang tính Embedding để quét Cache...")
        try:
            res = embedder.models.embed_content(
                model=os.getenv("model_embedding_name"),
                contents=current_query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            current_embedding = res.embeddings[0].values
        except Exception as e:
            print(f"Lỗi embedding lúc check cache: {e}")
            return None

        similarities = []
        for idx, item in enumerate(cache_copy):
            # Nếu bản ghi cũ chưa có embedding (lỗi dữ liệu cũ), bỏ qua
            if "embedding" not in item:
                continue
            sim = cosine_similarity(current_embedding, item["embedding"])
            similarities.append((idx, sim))

        if not similarities:
            return None

        # Sắp xếp giảm dần theo điểm Cosine và lấy Top 5
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_5_indices = [idx for idx, sim in similarities[:5]]

        # 3. Reranker (Lọc tinh Top 5)
        pairs = [[current_query, cache_copy[idx]["query"]] for idx in top_5_indices]

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128,
            )
            scores = (
                model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
            scores = torch.sigmoid(scores).tolist()

        if not isinstance(scores, list):
            scores = [scores]

        best_score = -1
        best_idx_in_pairs = -1
        for i, score in enumerate(scores):
            if score > best_score:
                best_score = score
                best_idx_in_pairs = i

        if best_score >= self.threshold:
            # print(f"Đã tìm thấy trong Cache (Vector + Rerank) - Score: {best_score:.2f}")
            real_idx = top_5_indices[best_idx_in_pairs]
            return cache_copy[real_idx]["answer"]

        return None

    def _add_async(self, query, answer):
        try:
            print("Đang lưu kết quả vào Cache ngầm...")
            res = embedder.models.embed_content(
                model=os.getenv("model_embedding_name"),
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            query_embedding = res.embeddings[0].values
            entry = {"query": query, "answer": answer, "embedding": query_embedding}

            with self.lock:
                # Kiểm tra lại xem trong lúc đợi tính embedding, query này đã có chưa
                if all(
                    item["query"].lower().strip() != query.lower().strip()
                    for item in self.cache
                ):
                    self.cache.append(entry)
                    # Nếu vượt quá mức trần, xóa phần tử đầu tiên (cũ nhất)
                    if len(self.cache) > self.max_size:
                        self.cache.pop(0)
                    self._save_cache_to_disk()
            print("Đã cập nhật Semantic Cache thành công!")
        except Exception as e:
            print(f"Lỗi khi add cache: {e}")

    def add(self, query, answer):
        if "Có lỗi xảy ra" in answer or "Xin lỗi" in answer:
            return
        # Chạy bất đồng bộ để không chặn luồng trả lời cho người dùng
        threading.Thread(target=self._add_async, args=(query, answer)).start()

    def flush(self):
        with self.lock:
            self.cache = []
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            print("Đã xóa sạch Semantic Cache!")


# Khởi tạo Singleton
semantic_cache = SemanticCache(max_size=20)
