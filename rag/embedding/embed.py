from dotenv import load_dotenv
import os
import json
import numpy as np  
load_dotenv('config/.env')
os.environ["HF_HOME"] = os.getenv("HF_HOME")
token= os.getenv('HF_TOKEN')

from google import genai
from google.genai import types
client = genai.Client(api_key=os.getenv('API_KEY'))

def load_embedder():
    return client  # genai client

def embedding():
    chunks = []
    with open('data/processed/data-cleaned.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line.strip()))

    texts = [chunk['text'] for chunk in chunks]

    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = client.models.embed_content(
            model=os.getenv('model_embedding_name'),
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        for emb in result.embeddings:
            all_embeddings.append(emb.values)
        print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.array(all_embeddings)
    np.save('data/vectors/vectors2.npy', embeddings)

    with open('data/vectors/vectors2.jsonl', 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps({"text": chunk['text']}, ensure_ascii=False) + '\n')

    print(f"\nSaved: vectors2.npy {embeddings.shape}")
    print(f"Saved: vectors2.jsonl ({len(chunks)} items)")
