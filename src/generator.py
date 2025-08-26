import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

def generate_response(context, query):
    """Generate response from LLM with given context."""
    prompt = f"""
You are an expert assistant for answering questions from PDFs. 
Use the given context (text + tables). 
If information is missing, say "Not found in the provided document."

Context:
{context}

Question:
{query}

Answer in a clear, structured way:
- If text → provide summary in sentences.
- If table → extract key rows/columns and explain briefly.
    """
    response = llm.invoke(prompt)
    return response.content
