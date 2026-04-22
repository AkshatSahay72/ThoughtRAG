import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.environ.get("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_respond(self, query: str, top_k: int = 5) -> dict:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = []
        sources = []
        for r in results:
            if r["metadata"]:
                texts.append(r["metadata"].get("text", ""))
                sources.append({
                    "doc": os.path.basename(r["metadata"].get("source", "unknown")),
                    "page": r["metadata"].get("page", 0)
                })
        
        context = "\n\n".join(texts)
        if not context:
            return {"answer": "No relevant documents found.", "sources": []}
            
        prompt = f"""Use the following context to answer the query: '{query}'\n\nContext:\n{context}\n\nAnswer:"""
        response = self.llm.invoke([prompt])
        
        return {
            "answer": response.content,
            "sources": sources
        }

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    result = rag_search.search_and_respond(query, top_k=3)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])