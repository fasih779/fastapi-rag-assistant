# 🤖 FastAPI RAG Assistant

> A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you chat with any codebase — powered by **Llama 3.3-70B**, **Qdrant Vector DB**, and **Sentence Transformers**, served via a **FastAPI** backend with a beautiful **Streamlit** frontend.

---

## ✨ Features

- 🔍 **Semantic Code Search** — Embeds your codebase with `all-MiniLM-L6-v2` and retrieves the most relevant chunks for every query
- 🦙 **Llama 3.3-70B via Groq** — Ultra-fast LLM inference for accurate, context-aware answers
- 📦 **Qdrant Vector Store** — Supports both local on-disk storage and Qdrant Cloud
- 🧩 **Smart Chunking** — Language-aware splitting for Python code and Markdown/RST documentation
- ⚡ **FastAPI Backend** — Clean REST API with `/query` and `/health` endpoints
- 🎨 **Streamlit Chat UI** — Polished chat interface with custom CSS styling
- 🔗 **LangChain Pipeline** — Full RAG chain with retriever, prompt template, and output parser

---

## 🏗️ Architecture

```
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│  Streamlit UI   │──────▶│  FastAPI Backend  │──────▶│   RAG Chain     │
│   (app.py)      │  HTTP │    (api.py)        │       │ (rag_chain.py)  │
└─────────────────┘       └──────────────────┘       └────────┬────────┘
                                                               │
                          ┌────────────────────────────────────┤
                          │                                    │
                 ┌────────▼────────┐                ┌─────────▼────────┐
                 │  Qdrant Vector  │                │  Groq / Llama    │
                 │     Store       │                │   3.3-70B LLM    │
                 │ (embedding.py)  │                └──────────────────┘
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │  Code Chunker   │
                 │  (chunker.py)   │
                 └─────────────────┘
```

---

## 📁 Project Structure

```
RAG Project/
├── app.py              # Streamlit chat frontend
├── api.py              # FastAPI REST API server
├── rag_chain.py        # LangChain RAG pipeline
├── embedding.py        # Embedding model + Qdrant vector store
├── chunker.py          # Document loader & language-aware chunker
├── style.css           # Custom Streamlit CSS styling
├── .streamlit/
│   └── config.toml     # Streamlit theme configuration
├── local_qdrant/       # Local Qdrant persistent storage (auto-generated)
├── fastapi/            # Source codebase to index (FastAPI library)
├── .env.example        # Environment variable template (copy to .env)
└── requirements.txt    # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/) (free tier available)
- *(Optional)* A [Qdrant Cloud](https://cloud.qdrant.io/) account for cloud vector storage

### 1. Clone the Repository

```bash
git clone https://github.com/fasih779/fastapi-rag-assistant.git
cd fastapi-rag-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here

# Optional — leave blank to use local on-disk Qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
```

### 5. Index Your Codebase

Place the source code you want to query inside the `fastapi/` folder (or change the path in `embedding.py`), then run:

```bash
python embedding.py
```

This will chunk, embed, and store all documents into Qdrant.

### 6. Start the FastAPI Backend

```bash
uvicorn api:app --port 8000 --reload
```

The API will be live at `http://localhost:8000`.  
Swagger docs available at `http://localhost:8000/docs`.

### 7. Launch the Streamlit Frontend

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` and start chatting! 🎉

---

## 🔌 API Reference

### `POST /query`

Ask a question about the indexed codebase.

**Request Body:**
```json
{
  "question": "How does FastAPI handle dependency injection?"
}
```

**Response:**
```json
{
  "answer": "FastAPI handles dependency injection using the Depends() function..."
}
```

### `GET /health`

Check API server status.

```json
{ "status": "ok" }
```

---

## 🛠️ Configuration

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Groq API key for Llama 3.3-70B inference |
| `QDRANT_URL` | ❌ No | Qdrant Cloud cluster URL (uses local storage if omitted) |
| `QDRANT_API_KEY` | ❌ No | Qdrant Cloud API key |

---

## 🧠 How It Works

1. **Indexing** — `chunker.py` loads `.py`, `.md`, and `.rst` files and splits them using language-aware chunkers (Python AST-aware for code, Markdown header splitter for docs).
2. **Embedding** — `embedding.py` uses `sentence-transformers/all-MiniLM-L6-v2` to embed each chunk into a 384-dimensional vector and stores it in Qdrant with HNSW indexing.
3. **Retrieval** — On each query, the top-8 most semantically similar chunks are retrieved from Qdrant.
4. **Generation** — Retrieved context + the user question are passed to Llama 3.3-70B via Groq, which generates a final English answer.
5. **UI** — The Streamlit frontend communicates with the FastAPI backend via HTTP POST.

---

## 📦 Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.3-70B (via Groq) |
| Vector DB | Qdrant (local or cloud) |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| RAG Framework | LangChain |
| API Server | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Language | Python 3.9+ |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Fasih** — [@fasih779](https://github.com/fasih779)

---

> ⭐ If you found this helpful, give it a star on GitHub!
