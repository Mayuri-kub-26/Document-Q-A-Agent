# Document Q&A Agent – RAG Pipeline

A production-style **Retrieval-Augmented Generation (RAG)** backend service.

- **REST API** (FastAPI) for document ingestion, Q&A, health checks
- **Vector search** (ChromaDB + sentence-transformers embeddings)
- **LLM integration** (OpenAI / Ollama / mock provider)
- **Dockerized** with structured JSON logging

## What is RAG? (in one paragraph)

RAG = "search first, then answer". Instead of asking an LLM to answer from memory (which hallucinates), we first **retrieve** the most relevant chunks from your own documents using vector similarity, then **feed** those chunks to the LLM as context, and ask it to answer **only from that context**. Result: answers grounded in your data, updatable without retraining.

## Architecture

```
┌──────────┐   POST /ingest/*   ┌──────────────┐
│  Client  │ ─────────────────► │  FastAPI     │
└──────────┘                    │   (main.py)  │
     │  POST /query             └──────┬───────┘
     │                                 │
     ▼                                 ▼
┌──────────────────────────────────────────────┐
│  RAG Pipeline (rag_pipeline.py)              │
│  1. Retriever.search()  -> top-k chunks      │
│  2. LLM.generate(question, chunks)           │
└────────────┬────────────────────┬────────────┘
             ▼                    ▼
      ┌─────────────┐      ┌──────────────┐
      │ VectorStore │      │ LLM Provider │
      │ (ChromaDB)  │      │ mock/openai/ │
      │ embeddings  │      │ ollama       │
      └─────────────┘      └──────────────┘
```

### Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, routes, middleware, lifespan |
| `config.py` | Settings from env / `.env` |
| `models.py` | Pydantic request/response schemas |
| `vector_store.py` | Persistent Chroma collection wrapper |
| `chunker.py` | Text chunking (sliding window) |
| `retriever.py` | Top-k retrieval |
| `ingestion.py` | Chunk + store documents |
| `llm.py` | LLM provider abstraction |
| `rag_pipeline.py` | Orchestration |
| `logging_config.py` | Structured JSON logging |
| `streamlit_app.py` | Streamlit web UI (chat, upload, store) |

## How to Run (local)

```bash
cd D:\Rag_Application
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open docs: http://localhost:8000/docs

### Streamlit frontend (recommended UI)

With the API running, start the UI in a second terminal:

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501. Use the **Upload documents** tab to add files, then the **Chat** tab to ask questions.

## How to Run (Docker)

```bash
docker compose up --build
```

API: http://localhost:8000 · Streamlit UI: http://localhost:8501

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + document count |
| POST | `/ingest/text` | Ingest raw text |
| POST | `/ingest/file` | Upload txt/md/pdf/docx |
| POST | `/query` | Ask a question |
| POST | `/reset` | Clear vector store |

### Example

```bash
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text":"Paris is the capital of France.","source":"france.txt"}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the capital of France?"}'
```

## Configuration

See `.env.example`. Key vars: `LLM_PROVIDER`, `OPENAI_API_KEY`, `CHUNK_SIZE`, `TOP_K`.

Supported free LLM providers (no paid key needed beyond free tier):
- `gemini` → Google Gemini (`GEMINI_API_KEY`)
- `openrouter` → OpenRouter free models (`OPENROUTER_API_KEY`)
- `ollama` → local Ollama
- `mock` → offline extractive (default, no key)

## Tests

```bash
pytest -q
```

## Notes

- Default LLM provider is `mock` so the service runs with no API key (useful for demos/tests). Set `LLM_PROVIDER=openai` (+ key) or `ollama` for real generation.
- Embeddings use `all-MiniLM-L6-v2` (local, ~90MB, downloaded on first run).
- Vector DB persisted under `data/chroma`.
