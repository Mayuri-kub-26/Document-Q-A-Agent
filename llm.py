"""LLM provider abstraction (mock / OpenAI-compatible / Ollama)."""

import os
from typing import Dict, List

from openai import OpenAI

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = (
    "You are a helpful, friendly AI assistant. Answer the user's question in natural, conversational language. "
    "Use the supplied reference material to ground your answer, but do NOT say phrases like "
    "'based on the provided context' or 'according to the context'. Just answer directly. "
    "Be concise but complete; use short paragraphs or bullet lists when helpful. "
    "Include inline citations like [1], [2] when you use a reference. "
    "If the references do not contain the answer, say you don't know and suggest a web search or follow-up. "
    "Use the conversation history to understand follow-up questions."
)


def _build_prompt(question: str, contexts: List[str], history: List[Dict[str, str]] = None) -> str:
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    history_text = ""
    if history:
        for turn in history[-8:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            history_text += f"{role}: {turn.get('content', '')}\n"
    parts = []
    if history_text:
        parts.append(f"Conversation so far:\n{history_text}")
    if ctx:
        parts.append(f"References:\n{ctx}")
    parts.append(f"Question: {question}\n\nAnswer:")
    return "\n\n".join(parts)


class BaseLLM:
    model_name: str = "unknown"

    def generate(self, question: str, contexts: List[str], history: List[Dict[str, str]] = None) -> str:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """Deterministic, offline LLM used for demos/tests (no API key needed)."""

    model_name = "mock-extractive"

    def generate(self, question: str, contexts: List[str], history: List[Dict[str, str]] = None) -> str:
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
    def __init__(self, base_url: str, api_key: str, model: str, extra_headers: dict = None) -> None:
        client_kwargs = {"base_url": base_url, "api_key": api_key or "dummy"}
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers
        self.client = OpenAI(**client_kwargs)
        self.model_name = model
        self._model = model

    def generate(self, question: str, contexts: List[str], history: List[Dict[str, str]] = None) -> str:
        prompt = _build_prompt(question, contexts, history)
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
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            if not content:
                # Some free/reasoning models put the answer in `reasoning`.
                reasoning = getattr(msg, "reasoning", None)
                if reasoning:
                    content = str(reasoning).strip()
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
    if provider == "gemini":
        if not settings.gemini_api_key:
            logger.warning("gemini_key_missing_falling_back_to_mock")
            return MockLLM()
        return OpenAICompatibleLLM(
            base_url=settings.gemini_base_url,
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )
    if provider == "openrouter":
        if not settings.openrouter_api_key:
            logger.warning("openrouter_key_missing_falling_back_to_mock")
            return MockLLM()
        return OpenAICompatibleLLM(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            extra_headers={
                "HTTP-Referer": "https://github.com/Mayuri-kub-26/Document-Q-A-Agent",
                "X-Title": "Document Q&A Agent",
            },
        )
    if provider == "ollama":
        return OpenAICompatibleLLM(
            base_url=settings.ollama_base_url,
            api_key="ollama",
            model=settings.ollama_model,
        )
    return MockLLM()
