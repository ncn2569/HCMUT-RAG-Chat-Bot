import re
import json
from rank_bm25 import BM25Okapi

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens

class BM25Retriever:
    def __init__(self,jsonl_path:str):
        self.document=[]
        with open(jsonl_path,'r',encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item=json.loads(line)
                    self.document.append(item['text'])
    
        tokenized_corpus =[preprocess_text(text) for text in self.document]
        self.bm25=BM25Okapi(tokenized_corpus)
    def search(self, query,top_k):
        tokenized_query=preprocess_text(query)
        doc_scores=self.bm25.get_scores(tokenized_query) # không lấy luôn chuỗi, để dung hợp vs rrf
        top_answers_bm25=doc_scores.argsort()[-top_k:][::-1]
        results=[(idx,rank+1) for rank,idx in enumerate(top_answers_bm25)]
        return results

