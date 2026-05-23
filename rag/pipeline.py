import json
import numpy as np
from dotenv import load_dotenv

import os
load_dotenv('config/.env')
hf_home= os.getenv("HF_HOME")
if hf_home is not None:
    os.environ["HF_HOME"] = hf_home # vấn đề bộ nhớ giá như có thêm tiền
from google import genai
client = genai.Client(api_key=os.getenv('API_KEY'))

from rag.retrieval.rerank import rerank
from rag.chat.history import clear_history
from rag.retrieval.hyde import generate_hypothetical_query
from rag.retrieval.dense_search import dense_search
from rag.retrieval.rrf_fuse import rrf_fuse
from rag.generation.build_prompt import build_prompt, rewrite_and_classify_query
from rag.embedding.embed import load_embedder
from rag.chat.history import get_history, add_turn, print_history
from rag.routing.query_router import classify_query
from rag.chat.semantic_cache import semantic_cache
from rag.retrieval.bm25 import BM25Retriever
embedder=load_embedder()
embeddings = np.load('data/vectors/vectors1.npy')
bm25=BM25Retriever('data/vectors/vectors1.jsonl')
data = []
with open('data/vectors/vectors1.jsonl', "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))  

def rag_query(current_query):
    # 1. Kiểm tra Semantic Cache để trả lời tức thì
    cached_answer = semantic_cache.check(current_query)
    if cached_answer:
        add_turn(current_query, cached_answer, current_query)
        return cached_answer

    history = get_history()
    # Rewrite and Classify query in 1 API call
    rewritten, query_type = rewrite_and_classify_query(current_query, history)
    if query_type == 'SIMPLE':
        dense_orig = dense_search(rewritten, embedder, embeddings, top_k=10)
        bm25_result=bm25.search(rewritten,top_k=10)
        rrf_list=rrf_fuse(dense_orig,bm25_result,k=60,weights=[0.5,1.0])
    elif query_type == 'COMPLEX':
        hyde_query = generate_hypothetical_query(rewritten)
        dense_orig = dense_search(rewritten, embedder, embeddings, top_k=10)
        dense_hyde = dense_search(hyde_query, embedder, embeddings, top_k=10)
        bm25_result=bm25.search(rewritten,top_k=10)
        rrf_list=rrf_fuse(dense_orig,dense_hyde,bm25_result,k=60,weights=[1.0,0.5,2.0])
    #rerank (cherry on the top)
    rerank_list = rrf_list[:15]
    final_list = rerank(rewritten, rerank_list,data, top_k=5)
    
    contexts = [data[idx]["text"] for idx, _ in final_list]

    prompt = build_prompt(rewritten, contexts)
    try:
        response = client.models.generate_content(
            model=os.getenv('model_name'),
            contents=prompt
        )

        answer = response.text.strip() 
    except Exception as e:
        print(f"Error: {e}")
        answer = "Có lỗi xảy ra r bạn oi, :)))) thử lại giúp mình sau nha, có thể là google đang nghẽn sever ấy mà hem sao đâu. Xí nữa hỏi lại nhen."
    add_turn(current_query, answer, rewritten)

    # 2. Lưu vào Cache (chỉ khi không có lỗi)
    if "Có lỗi xảy ra" not in answer:
        semantic_cache.add(current_query, answer)

    # print_history() #debugging
    return answer

def reset_history():
    """Reset lịch sử chat"""
    clear_history()


    
# hàm để test ragas, không cần bận tâm
def rag_query_test(current_query):
    history = get_history()
    # Rewrite and Classify query in 1 API call
    rewritten, query_type = rewrite_and_classify_query(current_query, history)
    if query_type == 'SIMPLE':
        dense_orig = dense_search(rewritten, embedder, embeddings, top_k=10)
        bm25_result=bm25.search(rewritten,top_k=10)
        rrf_list=rrf_fuse(dense_orig,bm25_result,k=60,weights=[0.5,1.0])
    elif query_type == 'COMPLEX':
        hyde_query = generate_hypothetical_query(rewritten)
        dense_orig = dense_search(rewritten, embedder, embeddings, top_k=10)
        dense_hyde = dense_search(hyde_query, embedder, embeddings, top_k=10)
        bm25_result=bm25.search(rewritten,top_k=10)
        rrf_list=rrf_fuse(dense_orig,dense_hyde,bm25_result,k=60,weights=[1.0,0.5,2.0])
    #OUT OF SCOPE:
    elif query_type == 'OUT_OF_SCOPE':
        answer= "Xin lỗi tôi chỉ trả lời những thông tin liên quan đến Trường Đại Học Bách Khoa Thành Phố Hồ Chí Minh, nếu có câu hỏi liên quan đến trường xin hãy cho tôi biết."
        add_turn(current_query, answer, rewritten)
        return answer
    #rerank (cherry on the top)
    rerank_list = rrf_list[:15]
    final_list = rerank(rewritten, rerank_list,data, top_k=5)
    
    contexts = [data[idx]["text"] for idx, _ in final_list]

    prompt = build_prompt(rewritten, contexts)
    try:
        response = client.models.generate_content(
            model=os.getenv('model_name'),
            contents=prompt
        )

        answer = response.text.strip() 
    except Exception as e:
        print(f"Error: {e}")
        answer = "Có lỗi xảy ra r bạn oi, :)))) thử lại giúp mình sau nha, có thể là google đang nghẽn sever ấy mà hem sao đâu. Xí nữa hỏi lại nhen."
    add_turn(current_query, answer, rewritten)

    # print_history() #debugging
    return answer,contexts