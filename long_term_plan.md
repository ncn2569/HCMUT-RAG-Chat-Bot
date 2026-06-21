# 🗺️ KẾ HOẠCH PHÁT TRIỂN: Nâng cấp ReAct Agentic RAG cho Tuyển sinh HCMUT

> **Profile**: Sinh viên năm 3 KHMT | RTX 1650 4GB | R5 4600H | Free tier only
> **Mục tiêu trước mắt**: Triển khai thành công Agentic RAG sử dụng mô hình ReAct (Reasoning and Acting) kết hợp Fallback Web Search, giữ vững scope tuyển sinh Bách Khoa TP.HCM.

---

## 🎯 Mục tiêu chính
Chuyển đổi pipeline RAG tuyến tính hiện tại thành một **AI Agent** tự suy luận và lập kế hoạch gọi công cụ (Tools) động:
1. Luôn ưu tiên tra cứu **Database nội bộ (90 câu)** trước.
2. Nếu database không có hoặc thông tin không đủ đáp ứng, tự động chuyển sang **Wikipedia / Web Search** để tìm câu trả lời bổ sung.
3. Ràng buộc chặt chẽ phạm vi câu hỏi và câu trả lời trong lĩnh vực tuyển sinh/thông tin trường ĐH Bách Khoa TP.HCM (HCMUT).

---

## ⚙️ Thiết kế Luồng ReAct Agent (Fallback Logic)

Thay vì viết code cứng (hardcode) nếu-thì, chúng ta sẽ để LLM tự quyết định dựa trên vòng lặp **Thought → Action → Observation**:

```text
                  [ User Query ] 
                        │
                        ▼
                ┌───────────────┐
                │  ReAct Agent  │◀──────────────────────┐
                └───────────────┘                       │
                        │                               │
             (Quyết định hành động)                     │
                        │                               │
            ┌───────────┴───────────┐                   │
            ▼                       ▼                   │
   [ search_hcmut_db ]    [ search_web_fallback ]       │
            │                       │                   │
     (Trả kết quả)             (Trả kết quả)            │
            ▼                       ▼                   │
     [ Observation ]         [ Observation ] ───────────┘
            │
            ├─► (Nếu đủ thông tin) ──► [ Final Answer ]
            └─► (Nếu thiếu) ─────────► (Lặp lại vòng suy luận)
```

---

## 🛑 BÀI TOÁN RATE LIMIT & GIẢI PHÁP TỐI ƯU (Gemini Flash Free API)

> [!WARNING]
> **Thách thức lớn nhất**: Gemini Free Tier giới hạn **15 RPM** (Requests Per Minute). 
> Nếu chạy ReAct dạng Prompt-based truyền thống (mỗi vòng lặp Thought -> Action -> Observation là một lượt gọi API), một câu hỏi của user có thể ngốn tới **3-4 API calls**, dễ gây nghẽn và sập chat giữa chừng.

### Giải pháp: Speculative Fallback Planning (Lập kế hoạch dự phòng suy đoán)
Thay vì bắt LLM chạy từng bước tuần tự (gọi API liên tục để xin ý kiến tiếp theo), chúng ta sẽ bắt LLM suy luận ra **Hành động chính** và **Hành động dự phòng** ngay trong 1 lượt gọi API đầu tiên bằng định dạng JSON.
*   **Số API call thực tế**: Tiết kiệm **tối đa chỉ 2 API calls** cho một câu hỏi phức tạp cần fallback, loại bỏ hoàn toàn rủi ro nghẽn 15 RPM do Agent chạy vòng lặp vô hạn (Infinite Loop).

---

## 📋 DỰ THẢO HIỆN THỰC SƠ BỘ (Draft Code)

### 1. Cấu trúc JSON Lập Kế Hoạch (LLM xuất ra)
```json
{
  "thought": "Người dùng hỏi điểm chuẩn ngành KHMT năm 2024. Tôi cần tra cứu Vector Database trước. Vì năm 2024 là thông tin mới, tôi sẽ chuẩn bị thêm câu truy vấn dự phòng trên web tuyển sinh.",
  "primary_action": {
    "tool": "search_hcmut_db",
    "query": "Điểm chuẩn ngành Khoa học Máy tính"
  },
  "fallback_action": {
    "tool": "search_web_fallback",
    "query": "Điểm chuẩn ngành Khoa học Máy tính Đại học Bách khoa TP.HCM 2024"
  }
}
```

### 2. File điều khiển Agent (`rag/agent.py`)
```python
import os
import json
import re
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv('API_KEY'))

def search_hcmut_db(query: str) -> str:
    """Gọi pipeline RAG hiện tại (Dense + Sparse + Rerank)"""
    # Trả về context tìm được, hoặc "Không tìm thấy thông tin"
    pass

def search_web_fallback(query: str) -> str:
    """Tra cứu web/wikipedia"""
    # Trả về text tóm tắt từ internet
    pass

def run_agent(user_query: str, chat_history: list) -> str:
    # --- BƯỚC 1: LLM LẬP KẾ HOẠCH (Call 1) ---
    planner_prompt = f"""
    Bạn là trợ lý tuyển sinh Đại học Bách Khoa TP.HCM. Hãy lập kế hoạch tìm thông tin cho câu hỏi: "{user_query}"
    
    Công cụ hiện có:
    - search_hcmut_db: Tìm trong database nội bộ của trường.
    - search_web_fallback: Tìm trên Wikipedia/Internet về tuyển sinh Bách Khoa.
    
    Hãy trả về định dạng JSON chính xác như sau:
    {{
      "thought": "Suy luận của bạn",
      "primary_action": {{
        "tool": "search_hcmut_db",
        "query": "câu truy vấn database"
      }},
      "fallback_action": {{
        "tool": "search_web_fallback",
        "query": "câu truy vấn internet"
      }}
    }}
    """
    
    response = client.models.generate_content(
        model=os.getenv('model_name'),
        contents=planner_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    
    plan = json.loads(response.text)
    
    # --- BƯỚC 2: PYTHON CHẠY TOOL THEO KẾ HOẠCH ---
    context = ""
    if plan.get("primary_action"):
        db_query = plan["primary_action"]["query"]
        context = search_hcmut_db(db_query)
        
    # Check xem DB có rỗng không, nếu rỗng -> chạy tool fallback
    if not context or "Không tìm thấy" in context:
        if plan.get("fallback_action"):
            web_query = plan["fallback_action"]["query"]
            # Ép thêm keyword để giữ đúng scope trường
            if "Bách Khoa" not in web_query and "HCMUT" not in web_query:
                web_query += " Đại học Bách Khoa TP.HCM"
            context = search_web_fallback(web_query)

    # --- BƯỚC 3: LLM ĐỌC CONTEXT VÀ TRẢ LỜI (Call 2) ---
    final_prompt = f"""
    Bạn là trợ lý tuyển sinh Bách Khoa TP.HCM. 
    Dựa vào thông tin thu thập được sau đây:
    {context}
    
    Hãy trả lời câu hỏi của người dùng: "{user_query}"
    """
    
    final_response = client.models.generate_content(
        model=os.getenv('model_name'),
        contents=final_prompt
    )
    
    return final_response.text
```

---

## 🛠️ Các Bước Triển Khai Thực Tế (Lộ trình tối giản để học tập)

*   **Bước 1**: Viết tool Wikipedia/Web Search trong `rag/tools/web_search.py`.
*   **Bước 2**: Refactor pipeline cũ trong `rag/pipeline.py` để tách hàm `search_hcmut_db`.
*   **Bước 3**: Tạo file `rag/agent.py` chứa vòng lặp Agent như thiết kế trên.
*   **Bước 4**: Tích hợp luồng Agent vào `app/streamlit_app.py` và hiển thị thought process lên Streamlit UI.
*   **Bước 5 (Nâng cao/Tùy chọn)**: Thêm bộ điều tiết `rate_limiter.py` nếu trong quá trình sử dụng thực tế gặp lỗi 429 do người dùng spam câu hỏi quá nhanh.
