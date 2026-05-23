
def classify_query(query: str, client, model_name: str) -> str:
    prompt = f"Phân loại câu hỏi sau: SIMPLE (câu hỏi đơn giản/ý định rõ ràng), COMPLEX (câu hỏi phức tạp/mơ hồ).\nCâu hỏi: {query}\nTrả lời 1 từ:"
    try:
        response= client.models.generate_content(
            model=model_name, 
            contents=prompt
            )
        answer=response.text.strip().upper()
    except Exception as e:
        print(f"error: {e}")
        answer='SIMPLE'
    return answer