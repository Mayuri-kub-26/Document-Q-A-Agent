"""Persistent Chroma vector store wrapper."""

import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Manages a persistent Chroma collection for document embeddings."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self.db_path = db_path or settings.chroma_path
        self.collection_name = collection_name or settings.collection_name
        self.embedding_model = embedding_model or settings.embedding_model

        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "vector_store_ready",
            extra={"db_path": self.db_path, "collection": self.collection_name, "count": self.count()},
        )

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not documents:
            return
        metadatas = metadatas or [{} for _ in documents]
        self.collection.add(documents=documents, ids=ids, metadatas=metadatas)
        logger.info("documents_added", extra={"count": len(documents)})

    def query(self, query: str, n_results: int = 4) -> List[Dict[str, Any]]:
        n = min(n_results, self.count()) if self.count() > 0 else n_results
        if n <= 0:
            return []
        try:
            res = self.collection.query(query_texts=[query], n_results=n)
        except Exception as exc:  # pragma: no cover
            logger.exception("vector_query_failed", extra={"error": str(exc)})
            return []

        docs = res.get("documents", [[]])[0]
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        out: List[Dict[str, Any]] = []
        for did, text, dist, meta in zip(ids, docs, dists, metas):
            # cosine distance -> similarity (1 - distance) for cosine space
            score = 1.0 - float(dist) if dist is not None else 0.0
            out.append(
                {
                    "id": did,
                    "text": text,
                    "score": round(score, 4),
                    "metadata": meta or {},
                }
            )
        return out

    def delete_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def save(self) -> None:
        # PersistentClient persists automatically; kept for API parity.
        pass
