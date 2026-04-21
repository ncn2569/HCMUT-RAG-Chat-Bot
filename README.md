---
title: HCMUT Chatbot
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---
# 🎓 HCMUT RAG Chatbot

Trợ lý tư vấn tuyển sinh thông minh cho Trường Đại học Bách khoa TP.HCM, sử dụng RAG (Retrieval-Augmented Generation) với Google Gemini.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)

## 📝 Tâm sự của tác giả 
- Dữ liệu hiện tại là từ một cá nhân rảnh rỗi iu trường (optional + seasonal) và có đam mê với AI. 
- Dữ liệu theo format Q-A, với 1 cột Question ghi các câu hỏi và 1 cột Answers ghi các câu trả lời (đây là dạng dễ crawl nhất). Tổng tất cả là 80 câu dòng (Tớ biết nó ít nhưng dòng nào cũng chan chứa mồ hôi hết).
- Cần API key Gemini để chạy (của tớ lúc làm thì tất nhiên là free rồi, svien bkhoa nghèo có tiếng mà).
- File .env và thư mục data/raw và data/processed đã được ignore nhưng vẫn còn data/vectors nếu bạn múa ngó thử về data mình đã cất công hái lượm.
- Tất nhiên là tương lai mình sẽ phát triển lên thêm (possibly agents nếu mình vô tình lượm đc một cái api key unlimited còn hiện tại thì chỉ là một Naive RAG bình thường thôi)

## 🚀 Tính năng

- **RAG Pipeline**: Kết hợp Dense Search + Sparse Search + HYDE  + RRF Fusion + Reranker
- **Chat History**: Quản lý ngữ cảnh hội thoại đa lượt
- **Query Rewriting**: Tự động viết lại câu hỏi dựa trên lịch sử
- **Gemini Integration**: Sử dụng Gemma 3 27B cho generation & Gemini Embedding 001 cho retrieval

## 📊 Đánh giá hệ thống (RAGAS Evaluation Metrics)

Hệ thống được đánh giá tự động thông qua framework **RAGAS**, các chỉ số gồm: (context precision + faithfulness + context recall + answer relevance).

Chấm bằng gemini 3 flash lite, thằng này thì nó đần hơn gemma 3 27b mình dùng để trả lời nhưng mà nó là con free duy nhất mà mình chạy mượt với ragas, mấy thằng khác thì nó không hỗ trợ hoặc lâu vl hoặc trả phí.

| Sample User Query | Faithfulness | Context Recall | Answer Relevancy | Context Precision |
| :--- | :---: | :---: | :---: | :---: |
| *Mã trường khi em đăng ký xét tuyển vào Bách khoa là gì?* | 1.0 | 1.0 | 0.7997 | 1 |
| *Em thuộc diện hộ nghèo là người dân tộc thiểu số, vậy có được miễn giảm học phí không và nộp hồ sơ ở đâu?* | 1.0 | 1.0 | 0.8669 | 1.0 |
| *Khoa Kỹ thuật Giao thông đào tạo những ngành nào vậy ạ?* | 1.0 | 1.0 | 0.8576 | 1.0 |
| *Ngành Khoa học Máy tính học những môn cơ sở ngành gì và sau khi ra trường có thể làm ở đâu?* | 1.0 | 1.0 | 0.8582 | 1.0 |
## 📁 Cấu trúc dự án
```text
hcmut-rag-chatbot/
├── app/
│   └── streamlit_app.py       # Giao diện web 
├── rag/
│   ├── ingestion/
│   │   └── chunking.py        # Xử lý chunking dữ liệu
│   ├── embedding/
│   │   └── embed.py           # Embedding với Gemini API
│   ├── retrieval/
│   │   ├── hyde.py            # Hypothetical Document Embedding
│   │   ├── dense_search.py    # Dense vector search
│   │   ├── bm25.py            # Sparse search using bm25
│   │   ├── rerank.py          # Reranker with bge-reranker-v2-m3
│   │   └── rrf_fuse.py        # Reciprocal Rank Fusion
│   ├── generation/
│   │   └── build_prompt.py    # Prompt engineering & query rewriting
│   ├── chat/
│   │   └── history.py         # Quản lý lịch sử hội thoại
│   └── pipeline.py            # RAG pipeline chính
├── data/
│   ├── raw/                   # Dữ liệu gốc (Excel)
│   ├── processed/             # Dữ liệu đã xử lý (JSONL)
│   └── vectors/               # Vector embeddings (NPY)
├── config/
│   └── .env                   # API keys (tự tạo với API_KEY theo template)
├── main.py                    
├── evaluation.py              # siêu phiền siêu lâu nếu không sở hữu api key có tiền (nghèo mà chịu thôi)   
├── report5.xlsx               # này là file đánh giá hiện tại.
├── Dockerfile                 
└── requirements.txt           
```
## 🛠️ Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/username/hcmut-rag-chatbot.git
cd hcmut-rag-chatbot
```
### 2. Tạo môi trường ảo
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```
### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```
### 4. Cấu hình biến môi trường

Tạo file `config/.env` với nội dung:

```env
# Google Gemini API
API_KEY="your-gemini-api-key-here"
model_name="gemma-3-27b-it"
model_embedding_name="gemini-embedding-001"

# Hugging Fac
HF_HOME=đường dẫn đến cache hugging face của bạn nếu không có thì sẽ dùng default.(mình thì ít ổ C nên cài vào ổ D)
HF_TOKEN= your huggings face token.
```
🔑 Lấy API key miễn phí tại: Google AI Studio

### 5. Chuẩn bị dữ liệu

**Bước 5.1:** Đặt file Excel vào thư mục `data/raw/`

**Bước 5.2:** Mở file `main.py`, sửa như sau để chạy embedding nếu có data mới:
```python
    # Trong main.py
    if __name__ == "__main__":
        from rag.embedding.embed import embedding
        embedding()
        # subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
```
**Bước 5.3:** Chạy lệnh để tạo vector embeddings:
```bash
    python main.py
```
**Bước 5.4:** Sau khi chạy xong, sửa lại `main.py` để chạy web:
```python
    # Trong main.py
    if __name__ == "__main__":
        from rag.embedding.embed import embedding
        #embedding()
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
```
**Bước 5.5:** Giờ chạy chatbot:
```bash
    python main.py
```
```text
⚙️ Cách hoạt động (Pipeline)
User Query → Query Rewriting (dựa trên history) 
    → HYDE (tạo hypothetical query)
    → Dense Search (2 queries: original + hyde)
    → Sparse Search 
    → RRF Fusion (k=60) ->Top 15 
    → Reranking -> Top 5 final
    → Build Prompt 
    → Generate Answer
    → Update History
```

