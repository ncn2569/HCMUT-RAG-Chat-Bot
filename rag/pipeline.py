import json
import numpy as np
from dotenv import load_dotenv

import os
load_dotenv('config/.env')
os.environ["HF_HOME"] = os.getenv("HF_HOME")

from google import genai
client = genai.Client(api_key=os.getenv('API_KEY'))
from rag.chat.history import clear_history
from rag.retrieval.hyde import generate_hypothetical_query
from rag.retrieval.dense_search import dense_search
from rag.retrieval.rrf_fuse import rrf_fuse,rrf_fuse_test
from rag.generation.build_prompt import build_prompt,rewrite_query_with_full_history
from rag.embedding.embed import load_embedder
from rag.chat.history import get_history, add_turn, print_history
from rag.retrieval.bm25 import BM25Retriever
embeddings = np.load('data/vectors/vectors2.npy')
embedder=load_embedder()
bm25=BM25Retriever('data/vectors/vectors2.jsonl')
data = []
with open('data/vectors/vectors2.jsonl', "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))  

def rag_query(current_query: str) -> str:
    history = get_history()
    # Step 1: Rewrite query gốc của người dùng để thành standalone query
    rewritten = rewrite_query_with_full_history(current_query, history)
    # Step 2: HYDE
    hyde_query = generate_hypothetical_query(rewritten)
    # Step 3: dense search 2 query (mỗi cái top 20)
    dense_orig = dense_search(rewritten, embedder, embeddings, top_k=10)
    dense_hyde = dense_search(hyde_query, embedder, embeddings, top_k=10)
    # Step 3.5: bm25 search
    bm25_result=bm25.search(rewritten,top_k=10)
    #fuse 2 kết quả với k = 60 để giảm mức độ chênh lệch score giữa top 1 và top 2
    final_fused=rrf_fuse_test(dense_orig,dense_hyde,bm25_result,k=60,weights=[1.0,0.5,2.5])
    # Lấy top 10 cho rerank 
    top_candidates = final_fused[:5]
    # Step 5: Rerank (Cross-Encoder) - Có thể bỏ qua vì overkill - nghiên cứu phát triển sau khi data lớn hơn
    # final_top = rerank(rewritten, top_candidates, top_k=3)
    final_top = top_candidates[:5]
    # Step 6: Generate
    contexts = [data[idx]["text"] for idx, _ in final_top]

    prompt = build_prompt(rewritten, contexts)

    response = client.models.generate_content(
        model=os.getenv('model_name'),
        contents=prompt
    )

    answer = response.text.strip()
    add_turn(current_query, answer, rewritten)
    # print_history() #debugging step
    return answer

def reset_history():
    """Reset lịch sử chat"""
    clear_history()


def rag_query_test(current_query: str) -> str:
    history = get_history()
    # Step 1: Rewrite query gốc của người dùng để thành standalone query
    rewritten = rewrite_query_with_full_history(current_query, history)
    # Step 2: HYDE
    hyde_query = generate_hypothetical_query(rewritten)
    # Step 3: dense search 2 query (mỗi cái top 20)
    dense_orig = dense_search(rewritten, embedder, embeddings, top_k=10)
    dense_hyde = dense_search(hyde_query, embedder, embeddings, top_k=10)
    # Step 3.5: bm25 search
    bm25_result=bm25.search(rewritten,top_k=10)
    #fuse 2 kết quả với k = 30 để giảm mức độ chênh lệch score giữa top1 và top 2
    # final_fused=rrf_fuse(dense_orig,dense_hyde,bm25_result,k=60)
    final_fused=rrf_fuse_test(dense_orig,dense_hyde,bm25_result,k=60,weights=[1.0,0.5,2.5])

    # Lấy top 10 cho rerank 
    top_candidates = final_fused[:5]
    # Step 5: Rerank (Cross-Encoder) - Có thể bỏ qua vì overkill - nghiên cứu phát triển sau khi data lớn hơn
    # final_top = rerank(rewritten, top_candidates, top_k=3)
    
    final_top = top_candidates[:5]
    # Step 6: Generate
    contexts = [data[idx]["text"] for idx, _ in final_top]

    prompt = build_prompt(rewritten, contexts)
    # print("testing",len(prompt))

    response = client.models.generate_content(
        model=os.getenv('model_name'),
        contents=prompt
    )

    answer = response.text.strip() if response.text else "Xin lỗi..."
    add_turn(current_query, answer, rewritten)
    # print_history() #debugging step
    return answer,contexts