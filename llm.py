"""LLM provider abstraction (mock / OpenAI-compatible / Ollama)."""

import os
from typing import List

from openai import OpenAI

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Always cite the source chunk ids you used when possible. Be concise."
)


def _build_prompt(question: str, contexts: List[str]) -> str:
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return (
        f"Context:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the context above:"
    )


class BaseLLM:
    model_name: str = "unknown"

    def generate(self, question: str, contexts: List[str]) -> str:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """Deterministic, offline LLM used for demos/tests (no API key needed)."""

    model_name = "mock-extractive"

    def generate(self, question: str, contexts: List[str]) -> str:
        if not contexts:
            return "I don't know — no relevant documents were found in the knowledge base."
        # Extractive-style answer: return the most relevant sentence-ish snippet.
        joined = " ".join(contexts)
        # Simple heuristic: echo best matching chunk trimmed.
        best = contexts[0].strip()
        if len(best) > 400:
            best = best[:400].rstrip() + "..."
        return (
            f"Based on the retrieved documents: {best}\n\n"
            "(This is a mock answer. Configure an LLM provider for generative answers.)"
        )


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key or "dummy")
        self.model_name = model
        self._model = model

    def generate(self, question: str, contexts: List[str]) -> str:
        prompt = _build_prompt(question, contexts)
        try:
            resp = self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                return "The model returned an empty response. Try rephrasing the question."
            return content
        except Exception as exc:
            logger.exception("llm_generate_failed", extra={"error": str(exc)})
            # Graceful fallback instead of crashing the request.
            return (
                f"[LLM provider error: {exc}]\n\n"
                "The LLM call failed, but retrieval succeeded. "
                "Check your API key / model name in .env, or switch LLM_PROVIDER=mock."
            )


def get_llm():
    provider = (settings.llm_provider or "mock").lower()
    if provider == "openai":
        if not settings.openai_api_key:
            logger.warning("openai_key_missing_falling_back_to_mock")
            return MockLLM()
        return OpenAICompatibleLLM(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
    if provider == "ollama":
        return OpenAICompatibleLLM(
            base_url=settings.ollama_base_url,
            api_key="ollama",
            model=settings.ollama_model,
        )
    return MockLLM()
