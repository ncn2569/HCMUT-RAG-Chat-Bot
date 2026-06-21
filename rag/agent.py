import json
import numpy as np
from dotenv import load_dotenv

import os
load_dotenv('config/.env')
hf_home= os.getenv("HF_HOME")
if hf_home is not None:
    os.environ["HF_HOME"] = hf_home # vấn đề bộ nhớ giá như có thêm tiền
tavily_key=os.getenv("tavily_key")
from google import genai
client = genai.Client(api_key=os.getenv('API_KEY'))

from pydantic import BaseModel, Field
from typing import List, Optional
from rag.generation.build_prompt import build_prompt
from rag.embedding.embed import load_embedder
from rag.retrieval.dense_search import dense_search
from rag.retrieval.rrf_fuse import rrf_fuse
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.rerank import rerank
embedder=load_embedder()
embeddings = np.load('data/vectors/vectors1.npy')
bm25=BM25Retriever('data/vectors/vectors1.jsonl')
data = []
with open('data/vectors/vectors1.jsonl', "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))  

def search_db(query):
    dense_orig = dense_search(query, embedder, embeddings, top_k=10)
    bm25_result=bm25.search(query,top_k=10)
    rrf_list=rrf_fuse(dense_orig,bm25_result,k=60,weights=[1.0,1.0])
    #rerank (cherry on the top)
    rerank_list = rrf_list[:15]
    final_list = rerank(query, rerank_list,data, top_k=5)
    
    contexts = [data[idx]["text"] for idx, _ in final_list]
    # final_list đã được sort giảm dần theo score trong hàm rerank
    confidence_score = final_list[0][1] if final_list else 0.0
    return contexts, confidence_score

def search_web(query):
    from tavily import TavilyClient
    client = TavilyClient(api_key=tavily_key)
    response = client.search(
        query,
        max_results=3,
        search_depth="basic",
        include_answer="basic"
    )
    contexts = []
    contexts.append(response["answer"])
    for res in response.get("results", []):
        contexts.append(f"Nguồn: {res['title']} - Nội dung: {res['content']}")
    return contexts

class Action(BaseModel):
    tools: str = Field(description="Công cụ cụ thể để hỗ trợ tìm kiếm thông tin (search_db hoặc search_web)")
    query: str = Field(description="Câu truy vấn cụ thể cho công cụ")
class Plan(BaseModel):
    thought: str = Field(description="Suy luận cụ thể để lập kế hoạch")
    primary_action: Action
    fallback_action: Optional[Action] = Field(default=None, description="Hành động dự phòng nếu hành động chính không có kết quả")

def run_agents(query,history):
    # Planning
    planner_prompt = f"""
    Bạn là trợ lý tuyển sinh Đại học Bách Khoa TP.HCM. 
    Hãy lập kế hoạch tìm thông tin cho câu hỏi: "{query}"

    Công cụ hiện có:
    - search_db(query): Tìm trong vector database nội bộ hiện tại về thông tin của trường BK để trả lời.
    - search_web(query): Tìm trên Web/Wikipedia về các thông tin và bối cảnh (thông tin mới năm 2024/2025/2026) để trả lời.
    """
    response = client.models.generate_content(
            model=os.getenv('model_name'),
            contents=planner_prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": Plan,
            }
        )
    plan = json.loads(response.text)

    # Debug
    print("\n" + "*" * 15 + " KẾ HOẠCH CỦA LLM " + "*" * 15)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    print("*" * 48)

    # Action
    contexts = []
    
    if plan.get("primary_action"):
        primary = plan["primary_action"]
        tool_name = primary["tools"]
        action_query = primary["query"]
        print(f" [DEBUG - TOOL] LLM quyết định dùng tool: '{tool_name}' với câu hỏi: '{action_query}'")
        
        if tool_name == "search_db":
            contexts, _ = search_db(action_query) 
        elif tool_name == "search_web":
            contexts = search_web(action_query)
        
    context_str = "\n\n".join(contexts) if contexts else "Không tìm thấy thông tin."
    final_prompt = build_prompt(query, context_str)
    
    final_response = client.models.generate_content(
        model=os.getenv('model_name'),
        contents=final_prompt
    )
    answer = final_response.text

    if "tôi không tìm thấy thông tin này trong cơ sở dữ liệu." in answer.lower():
        print(f" [DEBUG - REFLECTION] LLM báo không đủ thông tin từ nguồn chính, kích hoạt Fallback Web Search...")
        fallback = plan.get("fallback_action")
        
        web_query = fallback["query"] if fallback else query
        
        if "Bách Khoa" not in web_query and "HCMUT" not in web_query:
            web_query += " Đại học Bách Khoa TP.HCM" # back up info

        print(f" [DEBUG - TOOL] Gọi fallback search_web với câu hỏi: '{web_query}'")
        web_contexts = search_web(web_query)

        if web_contexts:
            # Ghép thêm dữ liệu web vào dữ liệu DB ban đầu
            contexts.extend(web_contexts)
            context_str = "\n\n".join(contexts)
            
            # Hỏi lại LLM lần 2 với lượng bối cảnh dồi dào hơn
            print(f" [DEBUG - RE-GENERATE] Bắt đầu tổng hợp lại câu trả lời với bối cảnh mới...")
            final_prompt = build_prompt(query, context_str)
            final_response = client.models.generate_content(
                model=os.getenv('model_name'),
                contents=final_prompt
            )
            answer = final_response.text
    return answer
