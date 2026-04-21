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
from rag.generation.build_prompt import build_prompt,rewrite_query_with_full_history
from rag.embedding.embed import load_embedder
from rag.chat.history import get_history, add_turn, print_history
from rag.retrieval.bm25 import BM25Retriever
embedder=load_embedder()
embeddings = np.load('data/vectors/vectors2.npy')
bm25=BM25Retriever('data/vectors/vectors2.jsonl')
data = []
with open('data/vectors/vectors2.jsonl', "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))  

def rag_query(current_query):
    history = get_history()
    #Rewrite query 
    rewritten = rewrite_query_with_full_history(current_query, history)
    # HYDE
    hyde_query = generate_hypothetical_query(rewritten)
    #dense search
    dense_orig = dense_search(rewritten, embedder, embeddings, top_k=10)
    dense_hyde = dense_search(hyde_query, embedder, embeddings, top_k=10)
    #bm25 search (mạnh vl)
    bm25_result=bm25.search(rewritten,top_k=10)
    #weights có thể bỏ nhưng không thích lắm
    rrf_list=rrf_fuse(dense_orig,dense_hyde,bm25_result,k=60,weights=[1.0,0.5,2.5])
    #rerank (cherry on the top)
    rerank_list = rrf_list[:15]
    final_list = rerank(rewritten, rerank_list,data, top_k=5)
    
    contexts = [data[idx]["text"] for idx, _ in final_list]

    prompt = build_prompt(rewritten, contexts)
    # print("testing",len(prompt))

    response = client.models.generate_content(
        model=os.getenv('model_name'),
        contents=prompt
    )

    answer = response.text.strip() 
    add_turn(current_query, answer, rewritten)

    # print_history() #debugging
    return answer

def reset_history():
    """Reset lịch sử chat"""
    clear_history()


    
# hàm để test ragas, không cần bận tâm
def rag_query_test(current_query):
    history = get_history()
    rewritten = rewrite_query_with_full_history(current_query, history)
    hyde_query = generate_hypothetical_query(rewritten)
    dense_orig = dense_search(rewritten, embedder, embeddings, top_k=10)
    dense_hyde = dense_search(hyde_query, embedder, embeddings, top_k=10)
    bm25_result=bm25.search(rewritten,top_k=10)
    rrf_list=rrf_fuse(dense_orig,dense_hyde,bm25_result,k=60,weights=[1.0,0.5,2.5])
    #rerank (cherry on the top)
    rerank_list = rrf_list[:15]
    final_list = rerank(rewritten, rerank_list,data, top_k=5)
    contexts = [data[idx]["text"] for idx, _ in final_list]
    prompt = build_prompt(rewritten, contexts)
    # print("testing",len(prompt))
    response = client.models.generate_content(
        model=os.getenv('model_name'),
        contents=prompt
    )
    answer = response.text.strip() 
    add_turn(current_query, answer, rewritten)
    # print_history() #debugging step
    return answer,contexts