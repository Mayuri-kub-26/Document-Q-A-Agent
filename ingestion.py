"""Document ingestion: chunk text and store in vector DB."""

import uuid
from typing import Any, Dict, List

from chunker import chunk_text
from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


def ingest_text(vector_store, text: str, source: str = "inline", metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    metadata = metadata or {}
    chunks = chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    if not chunks:
        return {"chunks_added": 0, "ids": []}

    ids = [f"{source}::{uuid.uuid4().hex[:8]}::{i}" for i in range(len(chunks))]
    metas = [{"source": source, "chunk_index": i, **metadata} for i in range(len(chunks))]

    vector_store.add_documents(documents=chunks, ids=ids, metadatas=metas)
    logger.info("ingest_done", extra={"source": source, "chunks": len(chunks)})

    return {"chunks_added": len(chunks), "ids": ids}
