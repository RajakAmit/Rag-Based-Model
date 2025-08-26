import numpy as np

def retrieve(query_emb, chunks, index, top_k=5):
    """Retrieve top-k relevant chunks."""
    D, I = index.search(np.array([query_emb]), top_k)
    return [chunks[i] for i in I[0]]
