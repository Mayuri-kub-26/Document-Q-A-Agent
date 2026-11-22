"""Retrieval helpers on top of the vector store."""

from typing import List

from vector_store import VectorStore
from logging_config import get_logger

logger = get_logger(__name__)


class DocumentRetriever:
    def __init__(self, vector_store: VectorStore, default_k: int = 4) -> None:
        self.vector_store = vector_store
        self.default_k = default_k

    def search(self, query: str, k: int = None) -> List[dict]:
        k = k or self.default_k
        results = self.vector_store.query(query=query, n_results=k)
        logger.info("retrieval_done", extra={"query": query[:80], "k": k, "hits": len(results)})
        return results
