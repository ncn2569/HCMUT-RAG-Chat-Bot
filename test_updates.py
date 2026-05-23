import os
from dotenv import load_dotenv

load_dotenv('config/.env')
from google import genai
client = genai.Client(api_key=os.getenv('API_KEY'))
model_name = os.getenv('model_name')

print("=== 1. TEST REWRITE AND CLASSIFY ===")
from rag.generation.build_prompt import rewrite_and_classify_query

# Test 1: No history, Simple question
query1 = "Học phí đại học Bách Khoa bao nhiêu?"
history1 = []
print(f"\nQuery: {query1}")
r1, c1 = rewrite_and_classify_query(query1, history1)
print(f"Rewritten: {r1}\nClass: {c1}")

# Test 2: With history, missing context
query2 = "Thế còn điểm chuẩn thì sao?"
history2 = [
    {"user": "Học phí đại học Bách Khoa bao nhiêu?", "rewritten": "Học phí đại học Bách Khoa bao nhiêu?", "assistant": "Khoảng 30 triệu."}
]
print(f"\nQuery: {query2}")
print(f"History: {history2}")
r2, c2 = rewrite_and_classify_query(query2, history2)
print(f"Rewritten: {r2}\nClass: {c2}")

# Test 3: Complex question
query3 = "Tại sao tôi nên chọn ngành Khoa học máy tính thay vì Kỹ thuật phần mềm?"
print(f"\nQuery: {query3}")
r3, c3 = rewrite_and_classify_query(query3, [])
print(f"Rewritten: {r3}\nClass: {c3}")


print("\n=== 2. TEST SEMANTIC CACHE ===")
from rag.chat.semantic_cache import semantic_cache
import time

# Thêm vào cache
print("Thêm vào cache: 'Học phí Bách Khoa bao nhiêu?' -> 'Khoảng 30 triệu/năm'")
semantic_cache.add("Học phí Bách Khoa bao nhiêu?", "Khoảng 30 triệu/năm")
time.sleep(2) # Chờ luồng ngầm chạy xong

# Test Exact Match
print("\nCheck Exact Match: 'Học phí Bách Khoa bao nhiêu?'")
ans = semantic_cache.check("Học phí Bách Khoa bao nhiêu?")
print(f"Kết quả: {ans}")

# Test Semantic Match
print("\nCheck Semantic Match: 'tiền học một học kỳ ở Bách khoa là bao nhiêu vậy?'")
ans2 = semantic_cache.check("tiền học một học kỳ ở Bách khoa là bao nhiêu vậy?")
print(f"Kết quả: {ans2}")

# Test Miss
print("\nCheck Miss: 'Ký túc xá ở đâu?'")
ans3 = semantic_cache.check("Ký túc xá ở đâu?")
print(f"Kết quả: {ans3}")

print("\n=== TEST HOÀN TẤT ===")
