"""RAG pipeline: retrieve relevant chunks, build context, generate answer."""

from typing import Any, Dict, List

from config import settings
from retriever import DocumentRetriever
from llm import get_llm, BaseLLM
from logging_config import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    def __init__(self, retriever: DocumentRetriever, llm=None, top_k: int = None) -> None:
        self.retriever = retriever
        self.llm: BaseLLM = llm or get_llm()
        self.top_k = top_k or settings.top_k

    def query(self, question: str, top_k: int = None, include_sources: bool = True):
        k = top_k or self.top_k
        retrieved = self.retriever.search(question, k=k)

        contexts = [r["text"] for r in retrieved]
        answer = self.llm.generate(question, contexts)

        # Groundedness check: if no retrieval, mark ungrounded.
        grounded = len(retrieved) > 0

        logger.info(
            "rag_query",
            extra={
                "question": question[:120],
                "hits": len(retrieved),
                "model": self.llm.model_name,
                "grounded": grounded,
            },
        )

        result: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "sources": retrieved if include_sources else [],
            "model": self.llm.model_name,
            "grounded": grounded,
        }
        return result
