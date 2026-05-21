# ⚡ QUICK WINS (Trước lúc thi) — Cải thiện Kỹ thuật RAG ngay lập tức

> **Mục tiêu**: Hoàn thiện pipeline hiện tại thành "Advanced Corrective RAG" trong 1-2 buổi tối để đưa ngay vào CV. Không đổi model, không cài thêm server, dùng nguyên setup hiện có (Gemini API + Python).

---

## 1. 🔍 Retrieval Metrics (Hit Rate & MRR) — Không tốn API

### Ý nghĩa (Nói trong phỏng vấn)
"Em đã tách riêng phần đánh giá Retrieval để debug. Nếu Hit Rate/MRR cao mà Answer Relevancy thấp → lỗi do prompt. Tính toán độc lập này giúp em biết BM25 và Dense đóng góp bao nhiêu."

### Thực hành (Code vào `evaluation_retrieval.py`)
Viết script chạy qua tập test 80 câu, tính xem trong top 5 docs trả về có chứa ground truth không (string matching hoặc keyword overlap).
- **Hit Rate@5**: Tỉ lệ phần trăm câu hỏi có ít nhất 1 doc đúng trong top 5.
- **MRR**: Trung bình của (1/vị trí_doc_đúng_đầu_tiên).

---

## 2. 🧠 Relevance Check (Corrective RAG)

### Ý nghĩa (Nói trong phỏng vấn)
"Sau khi lấy top 5 docs từ Reranker, em dùng LLM để filter lại một lần nữa (Relevance Check). Chỉ những docs thực sự chứa câu trả lời mới được đưa vào context. Việc này giảm nhiễu (noise) và chống LLM bịa chuyện."

### Thực hành (Code vào `rag/retrieval/relevance_check.py`)
```python
def check_relevance(query: str, contexts: list[str], client, model_name: str) -> list[str]:
    relevant = []
    for ctx in contexts:
        prompt = f"Văn bản sau có chứa thông tin trả lời câu hỏi '{query}' không? Trả lời YES hoặc NO.\n\nVăn bản: {ctx}"
        if "YES" in client.models.generate_content(model=model_name, contents=prompt).text.upper():
            relevant.append(ctx)
    return relevant if relevant else contexts[:2] # Fallback
```

---

## 3. 🏷️ Query Routing (Phân loại câu hỏi)

### Ý nghĩa (Nói trong phỏng vấn)
"Em đã thiết kế adaptive pipeline. Thay vì câu nào cũng chạy HyDE (tốn API, tốn thời gian), hệ thống route câu hỏi đơn giản đi đường thẳng (BM25 + Dense), câu phức tạp mới dùng HyDE, và reject ngay câu hỏi ngoài luồng."

### Thực hành (Code vào `rag/routing/query_router.py`)
```python
def classify_query(query: str, client, model_name: str) -> str:
    prompt = f"Phân loại câu hỏi sau: SIMPLE (tra cứu đơn giản), COMPLEX (cần suy luận/tổng hợp), OUT_OF_SCOPE (không thuộc ĐH Bách Khoa HCM).\nCâu hỏi: {query}\nTrả lời 1 từ:"
    return client.models.generate_content(model=model_name, contents=prompt).text.strip().upper()
```
Trong `pipeline.py`:
- `SIMPLE` → Không chạy HyDE, chỉ gọi `dense_search(query)` + `bm25`
- `COMPLEX` → Full HyDE pipeline
- `OUT_OF_SCOPE` → Trả lời "Xin lỗi, tôi chỉ hỗ trợ..."

---

## 4. 📈 Faithfulness Check (Chống Hallucination)

### Ý nghĩa (Nói trong phỏng vấn)
"Để tránh tuyệt đối việc chatbot 'bịa' thông tin tuyển sinh, em thêm lớp Faithfulness Verification sau generation. LLM đóng vai trò judge tự kiểm tra lại câu trả lời của chính nó dựa trên context."

### Thực hành (Code vào `rag/generation/hallucination_check.py`)
```python
def check_faithfulness(answer: str, contexts: list[str], client, model_name: str) -> bool:
    context_text = "\n".join(contexts)
    prompt = f"Câu trả lời sau có hoàn toàn dựa vào thông tin tham khảo không? Trả lời YES hoặc NO.\n\nTham khảo:\n{context_text}\n\nTrả lời:\n{answer}"
    return "YES" in client.models.generate_content(model=model_name, contents=prompt).text.upper()
```
Trong `pipeline.py`, nếu trả về NO, nối thêm câu: *"Lưu ý: Tôi không hoàn toàn chắc chắn về thông tin này, vui lòng đối chiếu lại."*

---

## 5. 📂 Data Quality — Đừng overkill RAG mà bỏ bê data

> **"Garbage in, garbage out."** Pipeline có xịn cỡ nào mà data dở thì kết quả vẫn dở. Đây là lời nhắc quan trọng nhất.

### Chẩn đoán data hiện tại

| Chỉ số | Giá trị | Nhận xét |
|:---|:---|:---|
| Tổng chunks | **80** | Khá ít — nhiều chủ đề chưa cover |
| Độ dài trung bình | **281 ký tự** | Ổn cho Q&A format |
| Chunk ngắn nhất | **71 ký tự** ("Mã trường là QSB") | Quá ngắn, embedding khó capture semantic |
| Chunk dài nhất | **1281 ký tự** (Học phí + Học bổng) | Quá dài, nhồi nhiều chủ đề vào 1 chunk |
| Metadata | **Không có** | Không có category, không có source |
| Format | **Q&A thuần** | Tốt cho retrieval nhưng thiếu đa dạng |

### 5 vấn đề cụ thể và cách fix

#### Vấn đề 1: Chunk quá ngắn — embedding bị "trống rỗng"
```
Hiện tại: "Q: Mã trường khi đăng ký tuyển sinh là gì\nA: Mã trường chúng tôi là QSB"
→ 71 ký tự. Embedding model khó biết đây là về "tuyển sinh đại học Bách khoa"
```
**Fix**: Thêm contextual prefix cho chunk ngắn:
```
Sửa thành: "Thông tin tuyển sinh Trường ĐH Bách khoa TP.HCM (HCMUT):
Q: Mã trường khi đăng ký tuyển sinh là gì
A: Mã trường chúng tôi là QSB"
```
→ Giúp embedding hiểu context tốt hơn. Đây chính là kỹ thuật **Contextual Chunking** (Anthropic, 2024).

#### Vấn đề 2: Chunk quá dài — nhồi nhiều fact vào 1 chunk
```
Hiện tại: 1 chunk dài 1281 ký tự chứa CẢ học phí LẪN học bổng LẪN chính sách giảm
→ User hỏi "học phí bao nhiêu?" → retrieve cả chunk → LLM phải đọc 1281 ký tự để tìm 1 con số
```
**Fix**: Tách chunk dài thành nhiều sub-chunks theo topic:
- Chunk A: Học phí chương trình đại trà
- Chunk B: Học phí chương trình tiên tiến
- Chunk C: Chính sách học bổng
- Chunk D: Chính sách giảm học phí

→ Retrieval chính xác hơn, context gọn hơn, LLM ít bịa hơn.

#### Vấn đề 3: Thiếu metadata — không filter được
```
Hiện tại: {"text": "Q: ... A: ..."}  ← chỉ có text, không biết thuộc chủ đề gì
```
**Fix**: Thêm trường `category` và `source`:
```json
{
  "text": "Q: Khoa KHMT có những ngành nào?\nA: ...",
  "category": "ĐÀO_TẠO",
  "faculty": "KHMT",
  "source": "website_hcmut_2025"
}
```
→ Sau này có thể dùng metadata để pre-filter trước khi search (VD: chỉ search trong category "HỌC_PHÍ" nếu user hỏi về tiền).

#### Vấn đề 4: Coverage — nhiều topic chưa có data
Xem lại xem 80 câu đã cover đủ chưa. Checklist gợi ý:

| Chủ đề | Có chưa? | Cần thêm? |
|:---|:---:|:---|
| Thông tin chung (tên, địa chỉ, mã trường) | ✅ | |
| Phương thức xét tuyển | ✅ | Thêm ví dụ cụ thể? |
| Điểm chuẩn từng ngành | ❓ | Cần data chi tiết cho ~30 ngành |
| Học phí theo từng chương trình | ❓ | Tách riêng đại trà / tiên tiến / quốc tế |
| Chỉ tiêu tuyển sinh 2025 | ❓ | Con số cụ thể cho từng ngành |
| Lịch trình tuyển sinh (deadline) | ❓ | Ngày nộp hồ sơ, ngày thi, ngày công bố |
| Ký túc xá, cơ sở vật chất | ✅ | |
| Đời sống sinh viên, CLB | ❓ | Câu hỏi hay gặp |
| Cơ hội việc làm sau tốt nghiệp | ❓ | Thông tin hấp dẫn tuyển sinh |

**Fix**: Bổ sung thêm 30-50 cặp Q&A nữa cho các chủ đề còn thiếu. Data tốt hơn → mọi component RAG đều tốt hơn theo.

#### Vấn đề 5: Đa dạng câu hỏi cho cùng 1 fact
```
Hiện tại: 1 fact = 1 câu hỏi. Nếu user hỏi khác cách thì retrieval có thể miss.
VD: "Mã trường là gì?" có trong data
    "Code đăng ký thi ĐGNL trường Bách Khoa?" → có thể miss
    "QSB là trường nào?" → chắc chắn miss
```
**Fix**: Cho mỗi fact quan trọng, viết thêm 2-3 dạng câu hỏi khác nhau (query augmentation/paraphrasing). Hoặc đơn giản hơn: thêm synonyms vào answer text.

### Phỏng vấn nói gì về Data?
> "Em nhận ra rằng cải thiện data quality cho ROI cao hơn cải thiện pipeline. Cụ thể em đã: (1) thêm contextual prefix cho chunk ngắn để giúp embedding, (2) tách chunk dài thành sub-topic chunks để retrieval chính xác hơn, (3) bổ sung metadata category cho filtered search, và (4) tăng coverage lên ~120 Q&A để cover đủ các chủ đề tuyển sinh."
>
> **Câu này thể hiện tư duy rất trưởng thành** — biết balance giữa engineering và data, không chạy theo kỹ thuật mù quáng.

### Thứ tự ưu tiên data improvements

| # | Việc | Effort | Impact |
|:---:|:---|:---:|:---:|
| 1 | **Tách chunk dài** (>800 chars) thành sub-chunks | Thấp | ⭐⭐⭐⭐⭐ |
| 2 | **Thêm contextual prefix** cho chunk ngắn (<120 chars) | Thấp | ⭐⭐⭐⭐ |
| 3 | **Bổ sung thêm 30-50 Q&A** cho topic còn thiếu | Trung bình | ⭐⭐⭐⭐⭐ |
| 4 | **Thêm metadata** (category, faculty) vào mỗi chunk | Trung bình | ⭐⭐⭐ |
| 5 | **Viết thêm dạng câu hỏi** cho mỗi fact quan trọng | Trung bình | ⭐⭐⭐ |

> [!TIP]
> Bước 1 và 2 có thể làm bằng tay trong 30 phút (chỉ có 7 chunk ngắn + 9 chunk dài). Bước 3 tốn thời gian hơn nhưng impact lớn nhất — hãy làm khi có thời gian rảnh.

---

## 🎯 Pipeline sau Quick Wins + Data improvements
```
[Data Layer]
  Chunks enriched: contextual prefix + metadata + sub-topic splitting
                          ↓
[Pipeline Layer]
  Query → 1. Classify (Router)
          ├─ OUT_OF_SCOPE → Reject
          ├─ SIMPLE → Dense + BM25 → Rerank
          └─ COMPLEX → HyDE → Dense + BM25 → Rerank
               ↓
          2. Relevance Check (Filter top 5)
               ↓
          Generate Answer
               ↓
          3. Faithfulness Check (Verification)
               ↓
          Return to User
```

**Kết hợp cải thiện cả Data + Pipeline = project vững chắc, không bị overkill lệch hướng!**
