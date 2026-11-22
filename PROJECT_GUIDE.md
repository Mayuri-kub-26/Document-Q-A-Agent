# Project Guide – Document Q&A Agent (RAG Pipeline)

A simple, complete walkthrough of the project: what it does, how to build it,
challenges (especially model selection), and interview-style Q&A.

---

## 1. What is this project? (Plain English)

Imagine you have 1000 company PDFs. A user asks: *"What is our refund policy?"*

- **Without RAG:** You ask ChatGPT. It guesses from internet training data → may be wrong/hallucinated.
- **With RAG:** The system first **searches your PDFs** for the refund policy, then hands those exact paragraphs to the LLM and says *"Answer using ONLY this."* → accurate, private, up-to-date.

So RAG = **Retrieve** relevant docs → **Augment** the prompt → **Generate** answer.

---

## 2. The full pipeline (step by step)

### A. Ingestion (offline / on upload)
1. **Load** document (txt/pdf/docx) → plain text.
2. **Chunk** text into ~500-char pieces with overlap (so context isn't cut mid-sentence).
3. **Embed** each chunk → vector (list of ~384 numbers) using `all-MiniLM-L6-v2`.
4. **Store** chunk text + vector + metadata in ChromaDB.

### B. Query (online)
1. User asks a question.
2. **Embed** the question → vector.
3. **Vector search** → top-k most similar chunks (cosine similarity).
4. **Build prompt**: system instructions + retrieved chunks + question.
5. **LLM generates** answer constrained to context.
6. Return answer + source chunks (for transparency/grounding check).

### C. Health / ops
- `/health` → status, version, provider, document count.
- Structured JSON logs for every request and pipeline step.
- Docker for portable deploy.

---

## 3. How to build it (roadmap)

1. **Pick stack**: FastAPI (API), ChromaDB (vector DB), sentence-transformers (embeddings), OpenAI/Ollama (LLM).
2. **Config**: pydantic-settings reading `.env` (provider, keys, chunk size, top_k).
3. **Vector store**: PersistentClient + collection with embedding function.
4. **Chunker**: sliding window with word-boundary snapping.
5. **Ingestion**: parse file → chunk → add to store.
6. **Retriever**: query store → top-k.
7. **LLM layer**: abstraction with mock/openai/ollama providers.
8. **RAG pipeline**: retrieve → generate → return sources + grounded flag.
9. **API**: FastAPI routes, middleware (request id, timing), exception handler.
10. **Logging**: JSON formatter.
11. **Docker**: Dockerfile + compose.
12. **Tests**: TestClient in-process.

---

## 4. Challenges faced (the "real world" part)

### 4.1 Model selection (biggest challenge)
Choosing the LLM is not "pick the biggest". Trade-offs:

| Concern | Options | Trade-off |
|---------|---------|-----------|
| Quality | GPT-4o, Claude | Best answers, cost + latency + data privacy |
| Cost | GPT-4o-mini, Llama 3.1 8B | Cheap, slightly less reasoning |
| Privacy | Ollama (local Llama/Qwen) | Data stays on-prem, needs GPU/RAM, slower |
| Latency | Smaller models | Faster, less nuanced |
| Context window | 8k vs 128k | More chunks fit → better grounding but more cost |
| Groundedness | Instruction-tuned models | Some models ignore "use only context" and still hallucinate |

**Key challenges:**
- **Hallucination despite context**: Even with RAG, LLMs sometimes invent. Mitigation: strong system prompt ("answer only from context, else say I don't know"), low temperature (0.1–0.3), return source chunks so user can verify, and a `grounded` flag when retrieval is empty.
- **Model ignores context (lost in the middle)**: LLMs pay less attention to middle chunks. Mitigation: keep top-k small (3–5), put most relevant first, or use re-ranking.
- **Embedding vs LLM mismatch**: Embedding model must match the language/domain. A generic MiniLM may miss domain jargon → use domain-tuned embeddings or fine-tune.
- **Cost control**: Sending 4 chunks × 500 chars per query adds up. Use caching for repeated queries, smaller chunks, and cheaper models for simple queries (routing).
- **Provider lock-in**: Abstract LLM behind an interface (done here) so swapping OpenAI ↔ Ollama is one env var.

### 4.2 Chunking
- Too big → dilutes relevance, exceeds context.
- Too small → loses surrounding context.
- Fixed char split can cut sentences → use recursive/token splitters in production.
- Overlap helps but increases storage.

### 4.3 Retrieval quality
- Vector search alone misses exact keyword matches (e.g., product codes). Hybrid = vector + BM25 (keyword) search.
- Need re-ranking (cross-encoder) for better ordering.
- Metadata filtering (by source/date) improves precision.

### 4.4 Evaluation
- Hard to measure "is the answer right?" → use RAGAS (faithfulness, answer relevancy, context precision/recall) or human eval.
- Build a golden Q&A set for regression testing.

### 4.5 Ops
- Cold start: loading embedding model ~30s. Pre-download in Docker image.
- Chroma persistence & backups.
- Rate limiting, auth, input size limits (file upload cap).
- Observability: request ids, latency, token usage.

---

## 5. Interview-style Q&A (how an interviewer asks)

**Q1: What is RAG and why use it?**
A: Retrieval-Augmented Generation. It grounds LLM answers in your own documents, reducing hallucination, keeping data private and updatable without retraining.

**Q2: Walk me through your pipeline.**
A: Ingest: parse → chunk → embed → store in Chroma. Query: embed question → top-k vector search → build prompt with chunks → LLM generate → return answer + sources + grounded flag.

**Q3: Why chunk documents?**
A: LLMs have context limits and attention dilutes over long text. Small focused chunks improve retrieval precision and fit in context.

**Q4: How do you prevent hallucination?**
A: Strong system prompt ("only use context, else say I don't know"), low temperature, small top-k, return sources for verification, grounded flag when no retrieval, and evaluation with RAGAS faithfulness.

**Q5: How did you choose the LLM?**
A: Based on quality/cost/latency/privacy trade-offs. Defaulted to a cheap model (gpt-4o-mini) for cost, mock for dev/test, Ollama for on-prem privacy. Abstracted provider so swap is config-only.

**Q6: What embedding model and why?**
A: `all-MiniLM-L6-v2` — small, fast, good general semantic similarity, runs locally (no API cost). For domain-specific (medical/legal), would use domain-tuned or larger models.

**Q7: Cosine similarity vs others?**
A: Cosine measures angle between vectors → good for text where magnitude (length) shouldn't matter. Euclidean also works; cosine is standard for normalized embeddings.

**Q8: How do you handle "no relevant docs"?**
A: If retrieval returns empty, set `grounded=false` and LLM (mock) says it doesn't know; real LLM instructed to say so.

**Q9: Scaling concerns?**
A: Embedding model cold start, vector DB persistence, caching frequent queries, async endpoints, horizontal scaling (stateless API + shared vector DB), rate limiting.

**Q10: How do you evaluate?**
A: Golden Q&A dataset; metrics: faithfulness (answer supported by context), answer relevancy, context precision/recall. Also latency and cost per query.

**Q11: Security?**
A: Input size limits, file type allowlist, no secrets in code (env vars), CORS config, request ids for audit, potential PII redaction.

**Q12: Docker details?**
A: Slim Python base, install deps, copy code, volume for `data/` persistence, healthcheck, env_file. Pre-download model to avoid cold start.

---

## 6. All sides of the project (cheat sheet)

- **Ingestion side**: file parsing, chunking, embedding, storage, dedup (future), metadata.
- **Retrieval side**: embedding query, vector search, top-k, scoring, (future) hybrid + rerank.
- **Generation side**: prompt building, LLM call, temperature, max tokens, safety prompt.
- **API side**: REST endpoints, validation (Pydantic), error handling, CORS, request logging, request id.
- **Data side**: Chroma persistence, collection management, reset.
- **Config side**: env-based settings, provider switching.
- **Ops side**: Docker, health check, structured logs, tests.
- **Quality side**: grounded flag, source return, evaluation plan.

---

## 7. Common pitfalls (to mention in interviews)

- Forgetting overlap → cut context → wrong answers.
- Using too high temperature → hallucination.
- Not returning sources → user can't verify.
- Chunk size not tuned → poor retrieval.
- No evaluation → can't tell if changes help.
- Hardcoding API keys → security risk.

---

## 8. Next improvements (shows growth mindset)

- Hybrid search (BM25 + vector) + cross-encoder reranking.
- Token-based chunking (tiktoken).
- Query rewriting / HyDE.
- Caching (Redis) for repeated queries.
- Auth + rate limiting.
- Async endpoints + background ingestion for large files.
- RAGAS evaluation harness in CI.
- Multi-tenant collections.
