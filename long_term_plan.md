# 🗺️ LONG-TERM ROADMAP: RAG → AI Agents (Sau khi thi xong)

> **Profile**: Sinh viên năm 3 KHMT | RTX 1650 4GB | R5 4600H | Free tier only
> **Mục tiêu**: Portfolio-grade project cho Gen AI Engineer intern

---

## 🧭 Tổng quan: 3 Phases dài hạn

Sau khi hoàn thiện **Quick Wins** (Corrective RAG) trước kỳ thi, dự án của bạn đã ở mức rất tốt. Tuy nhiên, để thực sự gây ấn tượng mạnh và dấn thân vào mảng AI Agent/GenAI Engineer, đây là kế hoạch cho thời gian rảnh rỗi sau thi:

```
      Hiện tại                 Mục tiêu
         │                        │
 Phase 1 (Sau thi)   Phase 2      │ Phase 3
 Self-hosted RAG    Agentic       │ Multi-Agent
  + Ablation         RAG          │  System
         │                        │
         ▼─────────▶──────────▶───▼
```

---

## 📦 Nguồn lực (Target: 100% Free)

| Role | Model | Chạy ở đâu | Resource |
|:---|:---|:---|:---|
| **Worker (Local)** | `qwen3:4b` via Ollama | GPU local | ~2.8 GB VRAM |
| **Orchestrator** | Gemini Flash free API | Google Cloud | 0 VRAM |
| **Embedding** | `Qwen3-Embedding-0.6B` | CPU local | ~1.2 GB RAM |
| **Reranker** | `bge-reranker-v2-m3` | CPU local | ~1.1 GB RAM |

---

# Phase 1: Self-hosted RAG + Ablation Study (~2 tuần)

> **Mục tiêu**: Tách khỏi phụ thuộc Google API cho các tác vụ đơn giản, tự host model trên máy, và có số liệu chứng minh (Ablation).

### 1. Hybrid Architecture
- **Cài đặt Ollama** và pull model `qwen3:4b`.
- **Đổi Embedding Model**: Cài `sentence-transformers`, dùng `Qwen3-Embedding-0.6B` và re-embed lại bộ data 80 câu.
- **Hybrid LLM Client**: Viết client để:
  - Tác vụ nhẹ (Classify, Relevance Check, RAG search đơn giản) → gọi Qwen3 local (Free, không tốn API).
  - Tác vụ phức tạp (Complex reasoning) → gọi Gemini Flash.

### 2. Ablation Study (Nghiên cứu cắt bỏ)
- **Thiết lập**: Chạy pipeline 6 lần với 6 config khác nhau (VD: Bỏ HyDE, Bỏ BM25, Không Rerank...).
- **Đánh giá**: Chạy RAGAS + Hit Rate trên 28 test cases (Dùng Kaggle T4 GPU để chạy evaluation nhanh).
- **Kết quả**: Ra được một cái bảng so sánh chính xác BM25 giúp tăng Context Recall bao nhiêu %, HyDE tăng latency bao nhiêu. Đây là **bằng chứng thép** khi đi phỏng vấn.

---

# Phase 2: Agentic RAG (~3 tuần)

> **Mục tiêu**: Chatbot không chỉ đọc database mà còn biết **hành động** và gọi hàm (function calling).

### 1. Xây dựng Công cụ (Tools)
- `search_knowledge_base`: Chức năng RAG đã có.
- `calculate_admission_score`: Viết Python function tính điểm ĐGNL + THPT + Học bạ theo công thức Bách Khoa.
- `search_web`: Dùng API miễn phí (DuckDuckGo) để tra cứu thông tin mới.

### 2. Thiết kế ReAct Agent Loop
- Dùng **Gemini Flash làm Orchestrator** vì cần khả năng suy luận mạnh.
- Agent sẽ hoạt động theo vòng lặp (Thought → Action → Observation):
  1. User hỏi: "Em 900 ĐGNL, 25 THPT thì đậu KHMT không?"
  2. Agent nghĩ: "Cần tính tổng điểm trước."
  3. Agent gọi tool `calculate_admission_score(900, 25)`.
  4. Hệ thống chạy Python function trả về "Tổng điểm: 77.4".
  5. Agent nghĩ: "Cần tra điểm chuẩn KHMT."
  6. Agent gọi tool `search_knowledge_base("điểm chuẩn KHMT")`.
  7. Hệ thống trả về "Điểm chuẩn: 84.16".
  8. Agent trả lời user: "Bạn thiếu điểm để vào KHMT..."

---

# Phase 3: Multi-Agent + Model Context Protocol (MCP) (~3 tuần)

> **Mục tiêu**: Cập nhật công nghệ mới nhất cuối năm 2024/đầu 2025.

### 1. Kiến trúc Multi-Agent
Chia hệ thống thành nhiều Agent nhỏ chuyên biệt thay vì 1 cục to:
- **Router Agent**: Phân loại người dùng.
- **RAG Agent (Local Qwen)**: Chuyên trả lời quy chế.
- **Math Agent (Python)**: Chuyên tính toán điểm số.

### 2. Tích hợp MCP (Model Context Protocol)
- Sử dụng `FastMCP` trong Python để biến các tools của bạn (RAG search, Calculator) thành một **MCP Server** chuẩn.
- Điểm ăn tiền: Chứng tỏ bạn am hiểu chuẩn kết nối AI mới nhất (do Anthropic khởi xướng), giúp bất kỳ AI nào (Claude Desktop, Cursor...) cũng kết nối được vào database Bách Khoa của bạn.

---

**🔥 Thành quả cuối cùng**: Một portfolio master-piece cho vị trí Gen AI Engineer Intern.
