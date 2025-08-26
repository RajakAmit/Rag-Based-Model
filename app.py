import os

import streamlit as st
from dotenv import load_dotenv

from src.embedder import build_vector_store
from src.extractor import extract_text_and_tables
from src.graph_builder import build_rag_graph

# Load API Key
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found. Add it in .env file.")
    st.stop()

st.set_page_config(page_title="PDF RAG QA with LangGraph", layout="wide")
st.title("📄 RAG-based PDF Q&A System (LangGraph + OpenAI)")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    os.makedirs("data", exist_ok=True)

    for pdf in uploaded_files:
        file_path = os.path.join("data", pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.read())
        chunks = extract_text_and_tables(file_path)
        all_chunks.extend(chunks)

    st.success(f"✅ Extracted {len(all_chunks)} chunks from {len(uploaded_files)} PDFs")

    index, chunks = build_vector_store(all_chunks)
    rag_graph = build_rag_graph(index, chunks)

    query = st.text_input("Ask a question:")
    if query:
        state = {"query": query}
        result = rag_graph.invoke(state)
        st.subheader("Answer:")
        st.write(result["answer"])

        with st.expander("Retrieved Context"):
            st.write(result["context"])
