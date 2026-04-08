# 🚀 ThoughtRAG

An AI-powered Retrieval-Augmented Generation (RAG) system that allows users to query knowledge from PDFs using semantic search + LLM.

---

## 🧠 Features
- PDF ingestion & processing
- Intelligent chunking
- Embeddings using SentenceTransformers
- Vector database (ChromaDB)
- Semantic retrieval
- LLM-based answer generation (Groq API)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/AkshatSahay72/ThoughtRAG
cd ThoughtRAG
```

---

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate   # Windows
```

---

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 4. Add Environment Variables

Create a `.env` file in root directory:

```env
GROQ_API_KEY=your_api_key_here
```

---

## 📂 IMPORTANT: Data & Vector Store (READ THIS)

⚠️ This repository does NOT include:
- PDF data files
- Vector database (Chroma/FAISS)

👉 You MUST add your own data.

---

### 📥 Step 1: Add PDFs

Create a folder:

```bash
data/pdf_files/
```

Add your PDFs inside it.

---

### 📊 Step 2: Delete Old Vector Store (if exists)

```bash
rm -rf data/vector_store
```

(Windows)
```bash
rmdir /s /q data\vector_store
```

---

### 🧠 Step 3: Run Pipeline

Run your main script:

```bash
python main.py
```

This will:
- Load PDFs
- Chunk text
- Generate embeddings
- Store in vector DB

---

## 🔍 Usage

Example query:

```python
rag_simple("What is consciousness?", rag_retriever, llm)
```

---

## ⚠️ Notes

- Ensure PDFs contain **text (not scanned images)**
- Queries must be **relevant to your dataset**
- Always rebuild vector DB after changing data

---

## 🧨 Common Mistakes

❌ Using random PDFs (low-quality results)  
❌ Not deleting old vector DB  
❌ Asking unrelated queries  

---

## 🛠 Tech Stack

- Python
- LangChain
- ChromaDB
- SentenceTransformers
- Groq LLM

---

## 📌 Future Improvements

- Better retrieval (reranking)
- UI (Streamlit / React)
- Multi-document reasoning
- Citation support

---

## 👨‍💻 Author
Akshat Sahay

---

## 📅 Generated on
2026-04-08
