from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        print(f"[DEBUG] Loaded {len(documents)} raw documents for chunking.")
        if documents and hasattr(documents[0], 'page_content'):
            sample_content = documents[0].page_content[:150].replace('\n', ' ')
            print(f"[DEBUG] Sample document content: '{sample_content}...'")

        # Safely filter documents to ensure we only process those with text content
        valid_docs = [doc for doc in documents if hasattr(doc, 'page_content') and doc.page_content.strip()]
        print(f"[DEBUG] Valid documents after filtering empty content: {len(valid_docs)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(valid_docs)
        print(f"[INFO] Split {len(valid_docs)} valid documents into {len(chunks)} chunks.")
        
        if len(chunks) == 0:
            raise ValueError(f"Chunking resulted in 0 chunks from {len(valid_docs)} valid documents. Documents might be empty or un-splittable.")
            
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        if not chunks:
            print("[WARNING] Skipping embedding generation: chunks list is empty.")
            return np.array([])
            
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)