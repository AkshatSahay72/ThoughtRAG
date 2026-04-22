import os
import shutil
import json
import gradio as gr
from src.search import RAGSearch
from langchain_community.document_loaders import PyPDFLoader

# Initialize RAG Search (Load existing or build initial)
print("[INFO] Initializing RAG Search...")
rag_search = RAGSearch()

def process_files(files):
    if not files:
        return json.dumps({"status": "error", "message": "No files uploaded."})
    
    UPLOAD_DIR = os.path.join("data", "pdf_files")
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to create upload directory: {e}"})
    
    new_docs = []
    for file_obj in files:
        filename = "unknown.pdf"
        try:
            filepath = file_obj.name
            if not os.path.exists(filepath):
                return json.dumps({"status": "error", "message": f"File does not exist: {filepath}"})
                
            filename = os.path.basename(filepath)
            dest_path = os.path.join(UPLOAD_DIR, filename)
            shutil.copy(filepath, dest_path)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"File saving failed for {filename}: {e}"})
        
        try:
            loader = PyPDFLoader(dest_path)
            loaded_docs = loader.load()
            # safeguard: skip empty content
            valid_docs = [doc for doc in loaded_docs if doc.page_content.strip()]
            if not valid_docs:
                print(f"[WARNING] Skipping {filename}: no text content found. Possibly empty or corrupted.")
                continue
            new_docs.extend(valid_docs)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"PDF loading failed for {filename}: {str(e)}"})
            
    if new_docs:
        print(f"[INFO] Adding {len(new_docs)} new docs to vector store...")
        try:
            rag_search.vectorstore.build_from_documents(new_docs)
            return json.dumps({"status": "success", "message": f"Successfully uploaded and processed {len(files)} files."})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Vector store insertion failed: {str(e)}"})
            
    return json.dumps({"status": "error", "message": "No valid text content found in uploaded documents."})

def chat_interface(query, history):
    if history is None:
        history = []
        
    cleaned_history = []
    # Safely unpack history elements to strict dictionary format
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            cleaned_history.append(msg)
        elif isinstance(msg, (tuple, list)) and len(msg) == 2:
            if msg[0]: cleaned_history.append({"role": "user", "content": str(msg[0])})
            if msg[1]: cleaned_history.append({"role": "assistant", "content": str(msg[1])})
            
    history = cleaned_history

    if not query.strip():
        return "", history
    
    # Get structured response from our backend
    result = rag_search.search_and_respond(query, top_k=3)
    answer = result.get("answer", "No answer found.")
    sources = result.get("sources", [])
    
    # Format the sources properly
    source_texts = []
    seen_sources = set()
    for s in sources:
        doc = s.get("doc", "unknown")
        page = s.get("page", 0)
        source_str = f"{doc} (Page {page})"
        if source_str not in seen_sources:
            seen_sources.add(source_str)
            source_texts.append(source_str)
            
    if source_texts:
        answer += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in source_texts)
        
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
        
    return "", history

# Build Gradio UI
with gr.Blocks(title="ThoughtRAG") as demo:
    gr.Markdown("# 🧠 ThoughtRAG System")
    gr.Markdown("Upload your documents to the knowledge base and chat with your context-aware assistant.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Document Management")
            file_upload = gr.File(label="Upload PDFs", file_count="multiple")
            upload_status = gr.Textbox(label="Status", interactive=False)
            
            file_upload.upload(
                fn=process_files, 
                inputs=[file_upload], 
                outputs=[upload_status]
            )
            
        with gr.Column(scale=2):
            gr.Markdown("### Chat Interface")
            chatbot = gr.Chatbot(label="Conversation History")
            query_input = gr.Textbox(
                label="Enter your query", 
                placeholder="E.g., What is attention mechanism?"
            )
            
            query_input.submit(
                fn=chat_interface, 
                inputs=[query_input, chatbot], 
                outputs=[query_input, chatbot]
            )
            gr.ClearButton([query_input, chatbot, upload_status, file_upload])

if __name__ == "__main__":
    demo.launch()