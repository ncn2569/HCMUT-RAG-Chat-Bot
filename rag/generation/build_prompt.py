from google import genai
from dotenv import load_dotenv
import os
load_dotenv('config/.env')
client = genai.Client(api_key=os.getenv('API_KEY'))

def build_prompt(query: str, contexts: list) -> str:
    # Gộp top 3 QA thành context
    context_text = "\n\n".join([
        f"{text}"
        for text in contexts  
    ])

    prompt = f"""Bạn là trợ lý tư vấn tuyển sinh của Trường Đại học Bách khoa TP.HCM (HCMUT).

    Thông tin tham khảo:
    {context_text}

    Câu hỏi: {query}

    Hướng dẫn trả lời:
    - Chỉ sử dụng thông tin tham khảo nếu nó trực tiếp trả lời được câu hỏi.
    - Nếu hoàn toàn không có thông tin, trả lời: "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu."
    - TUYỆT ĐỐI Không bịa đặt thông tin.
    - Trả lời ngắn gọn, rõ ràng và trả lời đúng trọng tâm câu hỏi.
    - Nếu câu hỏi hoặc yêu cầu của người dùng không liên quan đến Trường Đại Học Bách Khoa Thành Phố Hồ Chí Minh thì trả lời "Xin lỗi tôi chỉ trả lời những thông tin liên quan đến Trường Đại Học Bách Khoa Thành Phố Hồ Chí Minh, nếu có câu hỏi liên quan đến trường xin hãy cho tôi biết.".
    Trả lời:"""
    return prompt

def rewrite_query_with_full_history(current_query: str, history: list) -> str:
    if not history:
        return current_query

    # Toàn bộ history
    history_text = "\n\n".join([
        f"User: {turn.get('rewritten', turn['user'])}\nAssistant: {turn['assistant']}"
        for turn in history[-5:]  # dùng rewritten nếu có, fallback về user gốc
    ])

    rewrite_prompt = f"""Dựa trên lịch sử trò chuyện gần nhất, viết lại câu hỏi sau thành câu độc lập, đầy đủ ngữ cảnh.

    Lịch sử trò chuyện:
    {history_text}

    Câu hỏi mới: "{current_query}"

    Hướng dẫn:
    - Nếu câu hỏi có đại từ hoặc thiếu ngữ cảnh, bổ sung từ lịch sử
    - Giữ nguyên ý nghĩa gốc, không thay đổi ý nghĩa của câu hỏi,
    - Chỉ viết lại câu hỏi, TUYỆT ĐỐI KHÔNG trả lời.

    Câu hỏi đã viết lại:"""

    try:
        response = client.models.generate_content(
            model=os.getenv('model_name'),
            contents=rewrite_prompt
        )
        rewritten = response.text.strip() if response.text else current_query
        rewritten = rewritten.strip('"').strip("'")
    except Exception as e:
        print(f"Error ở bước rewrite: {e}")
        rewritten = current_query 
        
    return rewritten

def rewrite_and_classify_query(current_query: str, history: list) -> tuple[str, str]:
    history_text = "Không có"
    if history:
        history_text = "\n\n".join([
            f"User: {turn.get('rewritten', turn['user'])}\nAssistant: {turn['assistant']}"
            for turn in history[-5:]
        ])

    prompt = f"""Dựa trên lịch sử trò chuyện (nếu có), hãy thực hiện 2 nhiệm vụ:
1. Viết lại câu hỏi sau thành câu độc lập, đầy đủ ngữ cảnh (nếu có đại từ chỉ định như 'nó', 'trường này'..., hãy thay thế bằng danh từ cụ thể từ lịch sử). Nếu không cần viết lại, giữ nguyên câu hỏi.
2. Phân loại câu hỏi ĐÃ VIẾT LẠI thành 1 trong 2 loại: SIMPLE (câu hỏi đơn giản/tra cứu thông tin trực tiếp), COMPLEX (câu hỏi phức tạp/cần suy luận/mơ hồ).

Lịch sử trò chuyện:
{history_text}

Câu hỏi hiện tại: "{current_query}"

Bạn PHẢI trả về định dạng JSON chính xác như sau, không xuất thêm bất kỳ chữ nào khác:
{{
  "rewritten_query": "câu hỏi đã viết lại",
  "query_type": "SIMPLE"
}}"""

    #try:
    response = client.models.generate_content(
        model=os.getenv('model_name'),
        contents=prompt
    )
    text = response.text.strip()
    import re
    import json
    
    # Sử dụng regex để tìm khối JSON phòng khi model trả về các câu rào trước/sau
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
        
    data = json.loads(text)
    rewritten = data.get("rewritten_query", current_query)
    q_type = data.get("query_type", "SIMPLE").upper()
    # if q_type not in ["SIMPLE", "COMPLEX"]:
    #     q_type = "SIMPLE"
    return rewritten, q_type
    # except Exception as e:
    #     print(f"Error ở bước rewrite & classify: {e}")
    #     return current_query, "SIMPLE"
