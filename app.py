from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    print("[INFO] Initializing RAG Search...")
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    print(f"[INFO] Running query: {query}")
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)