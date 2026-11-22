"""FastAPI application entry point for the Document Q&A (RAG) service."""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from logging_config import setup_logging, get_logger
from models import (
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from vector_store import VectorStore
from retriever import DocumentRetriever
from rag_pipeline import RAGPipeline
from ingestion import ingest_text

setup_logging()
logger = get_logger(__name__)

# Shared state (initialized on startup).
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", extra={"version": settings.app_version, "provider": settings.llm_provider})
    vs = VectorStore()
    retriever = DocumentRetriever(vs, default_k=settings.top_k)
    rag = RAGPipeline(retriever=retriever)
    _state["vector_store"] = vs
    _state["retriever"] = retriever
    _state["rag"] = rag
    yield
    logger.info("shutdown")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="REST API for document ingestion, RAG-based Q&A, and health checks.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex)
    request.state.request_id = rid
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 2)
    logger.info(
        "http_request",
        extra={
            "request_id": rid,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    response.headers["x-request-id"] = rid
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "-")
    logger.exception("unhandled_error", extra={"request_id": rid, "error": str(exc)})
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": rid})


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    vs = _state["vector_store"]
    rag = _state["rag"]
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        provider=rag.llm.model_name,
        documents=vs.count(),
    )


@app.get("/v1/models", tags=["system"])
def list_models():
    """OpenAI-compatible model listing (for clients that probe this endpoint)."""
    rag = _state.get("rag")
    model = rag.llm.model_name if rag else settings.openai_model
    return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": settings.llm_provider}]}


@app.post("/ingest/text", response_model=IngestResponse, tags=["ingestion"])
def ingest_text_endpoint(payload: IngestTextRequest):
    vs = _state["vector_store"]
    result = ingest_text(
        vector_store=vs,
        text=payload.text,
        source=payload.source or "inline",
        metadata=payload.metadata,
    )
    if result["chunks_added"] == 0:
        raise HTTPException(status_code=400, detail="No text content to ingest")
    return IngestResponse(
        document_id=result["ids"][0].split("::")[0] if result["ids"] else "inline",
        chunks_added=result["chunks_added"],
        source=payload.source or "inline",
    )


@app.post("/ingest/file", response_model=IngestResponse, tags=["ingestion"])
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    source: Optional[str] = Form(default=None),
):
    vs = _state["vector_store"]
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb}MB")

    name = file.filename.lower()
    text = ""
    if name.endswith(".txt") or name.endswith(".md"):
        text = content.decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        try:
            import io
            import pdfplumber

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}")
    elif name.endswith(".docx"):
        try:
            import io
            import docx

            doc = docx.Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse DOCX: {exc}")
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type (txt, md, pdf, docx)")

    src = source or file.filename
    result = ingest_text(vector_store=vs, text=text, source=src)
    if result["chunks_added"] == 0:
        raise HTTPException(status_code=400, detail="No extractable text in file")
    return IngestResponse(document_id=src, chunks_added=result["chunks_added"], source=src)


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query_endpoint(payload: QueryRequest):
    rag = _state["rag"]
    vs = _state["vector_store"]
    # Allow per-request override of top_k by rebuilding retriever view.
    if payload.top_k:
        from retriever import DocumentRetriever

        retriever = DocumentRetriever(vs, default_k=payload.top_k)
        rag.retriever = retriever

    result = rag.query(
        question=payload.question,
        top_k=payload.top_k,
        include_sources=payload.include_sources,
    )

    sources = [
        SourceChunk(
            id=s["id"],
            text=s["text"],
            score=s["score"],
            metadata=s.get("metadata", {}),
        )
        for s in result["sources"]
    ]
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
        model=result["model"],
        grounded=result["grounded"],
    )


@app.post("/reset", tags=["system"])
def reset_endpoint():
    vs = _state["vector_store"]
    vs.delete_collection()
    return {"status": "reset", "documents": vs.count()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
