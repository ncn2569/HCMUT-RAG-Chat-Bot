import re
from google import genai
from dotenv import load_dotenv
import os

load_dotenv("config/.env")
client = genai.Client(api_key=os.getenv("API_KEY"))


def generate_hypothetical_query(query, model_name=os.getenv("model_name")):

    prompt = f"""Bạn đang hỗ trợ hệ thống tìm kiếm trong cơ sở dữ liệu hỏi đáp.

    Hãy viết lại câu sau bằng cách diễn đạt khác nhưng giữ nguyên ý nghĩa.
    Mục tiêu là tạo một câu hỏi tương tự về mặt ngữ nghĩa để giúp tìm được các câu hỏi và tài liệu liên quan.

    Hướng dẫn:
    - Diễn đạt lại tự nhiên bằng cách khác.
    - Có thể thay đổi từ ngữ hoặc cấu trúc câu.
    - Chỉ trả về câu hỏi đã viết lại, không giải thích.

    Câu hỏi gốc: {query}

    Câu hỏi tương tự:"""

    try:
        response = client.models.generate_content(
            model=os.getenv("model_name"), contents=prompt
        )
        hypothetical = response.text.strip() if response.text else query
        hypothetical = hypothetical.split("\n")[0].strip()
        hypothetical = re.sub(r"^(Câu hỏi tương tự[:：]\s*)", "", hypothetical)
    except Exception as e:
        print(f"Error ở bước Hyde: {e}")
        hypothetical = query

    return hypothetical


def generate_hypothetical_document(query, model_name=os.getenv("model_name")):

    prompt = f"""Bạn đang hỗ trợ hệ thống tìm kiếm tài liệu.

    Hãy đóng vai một chuyên gia và viết một đoạn văn bản ngắn (khoảng 2-3 câu) trực tiếp trả lời hoặc cung cấp thông tin liên quan cho câu hỏi dưới đây.
    Mục tiêu là tạo ra một "tài liệu giả định" (hypothetical document) chứa các từ khóa và ngữ cảnh tự nhiên có khả năng xuất hiện trong tài liệu thật trong cơ sở dữ liệu.

    Hướng dẫn:
    - Trả lời trực tiếp, dạng văn trần thuật cung cấp thông tin (không viết lại câu hỏi).
    - Dùng từ vựng và văn phong trang trọng, học thuật liên quan đến ngữ cảnh trường đại học.
    - Chỉ trả về nội dung đoạn văn, tuyệt đối không giải thích thêm.

    Câu hỏi gốc: {query}

    Tài liệu giả định:"""

    try:
        response = client.models.generate_content(
            model=os.getenv("model_name"), contents=prompt
        )
        hypothetical = response.text.strip() if response.text else query
        hypothetical = re.sub(
            r"^(Tài liệu giả định[:：]\s*)", "", hypothetical, flags=re.IGNORECASE
        )
    except Exception as e:
        print(f"Error ở bước Hyde (Document): {e}")
        hypothetical = query

    return hypothetical
