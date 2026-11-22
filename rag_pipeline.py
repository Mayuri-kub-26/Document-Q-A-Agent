"""RAG pipeline: retrieve relevant chunks, build context, generate answer."""

from typing import Any, Dict, List, Optional

from config import settings
from retriever import DocumentRetriever
from llm import get_llm, BaseLLM
from logging_config import get_logger
from web_search import search_web

logger = get_logger(__name__)


class RAGPipeline:
    def __init__(self, retriever: DocumentRetriever, llm=None, top_k: int = None) -> None:
        self.retriever = retriever
        self.llm: BaseLLM = llm or get_llm()
        self.top_k = top_k or settings.top_k

    def query(
        self,
        question: str,
        top_k: int = None,
        include_sources: bool = True,
        use_web: bool = None,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        k = top_k or self.top_k
        retrieved = self.retriever.search(question, k=k)

        if use_web is None:
            use_web = settings.enable_web_search
        if use_web:
            retrieved = retrieved + search_web(question, settings.web_search_results)

        contexts = [r["text"] for r in retrieved]
        if contexts:
            answer = self.llm.generate(question, contexts, history=history)
        else:
            answer = (
                "I don't have enough information to answer that yet. "
                "Upload a document or enable live web search, then try again."
            )

        # Groundedness check: local docs or live web results count as grounding.
        grounded = len(retrieved) > 0

        logger.info(
            "rag_query",
            extra={
                "question": question[:120],
                "hits": len(retrieved),
                "model": self.llm.model_name,
                "grounded": grounded,
                "web": use_web,
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
