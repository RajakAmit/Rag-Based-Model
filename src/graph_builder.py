from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from src.embedder import embed_query
from src.generator import generate_response
from src.retriever import retrieve


class GraphState(TypedDict):
    query: str
    query_emb: List[float]
    context: str
    answer: str

def build_rag_graph(index, chunks):
    graph = StateGraph(GraphState)

    def embed_node(state: GraphState):
        q_emb = embed_query(state["query"])
        return {**state, "query_emb": q_emb}

    def retrieve_node(state: GraphState):
        retrieved_chunks = retrieve(state["query_emb"], chunks, index, top_k=4)
        context = "\n".join(retrieved_chunks)
        return {**state, "context": context}

    def generate_node(state: GraphState):
        answer = generate_response(state["context"], state["query"])
        return {**state, "answer": answer}

    graph.add_node("embed", embed_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("embed")
    graph.add_edge("embed", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
