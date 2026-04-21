import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3') #first recommend from gemini (AI sẽ thay thế con người sớm thôi)
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
model.eval()

def rerank(query, rrf_list, data, top_k=5):
    pairs = []
    for idx, _ in rrf_list:
        doc_text = data[idx]["text"]
        pairs.append([query, doc_text])

    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = torch.sigmoid(scores).tolist()


    # print(rrf_list,inputs,scores)
    results = []

    for i, (idx, _) in enumerate(rrf_list):
        score_val = scores[i] if isinstance(scores, list) else scores
        results.append((idx, score_val))

    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]