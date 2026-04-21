import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from google.genai import types
def dense_search(query, embedder, embeddings, top_k= 20):
 
    result = embedder.models.embed_content(
        model=os.getenv('model_embedding_name'),
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    vector = np.array(result.embeddings[0].values).reshape(1, -1)
    scores_original = cosine_similarity(vector, embeddings)[0]
    top_orig = scores_original.argsort()[-top_k:][::-1]

    res_orig = [(int(i), r + 1) for r, i in enumerate(top_orig)]
    
    return res_orig