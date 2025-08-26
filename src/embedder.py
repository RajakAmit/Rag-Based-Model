import os

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env")

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)

def build_vector_store(chunks):
    """Build FAISS index with embeddings."""
    vectors = embeddings_model.embed_documents(chunks)
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))
    return index, chunks

def embed_query(query):
    """Embed query for retrieval."""
    return np.array(embeddings_model.embed_query(query)).astype("float32")
