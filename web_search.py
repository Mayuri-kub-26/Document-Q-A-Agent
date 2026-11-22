"""Lightweight web search fallback for live / real-time information."""

from typing import Any, Dict, List

from duckduckgo_search import DDGS

from logging_config import get_logger

logger = get_logger(__name__)


def search_web(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Return web results as source-chunk-like dicts."""
    raw: List[Dict[str, Any]] = []
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
            if raw:
                break
        except Exception as exc:
            logger.warning("web_search_attempt_failed", extra={"attempt": attempt, "error": str(exc)})

    if not raw:
        logger.warning("web_search_no_results", extra={"query": query[:120]})
        return []

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(raw, 1):
        title = item.get("title") or "Web result"
        url = item.get("href") or ""
        body = item.get("body") or ""
        text = f"{title}\n{body}".strip()
        out.append(
            {
                "id": f"web::{i}",
                "text": text,
                "score": 0.0,
                "metadata": {
                    "type": "web",
                    "title": title,
                    "url": url,
                    "source": url or title,
                },
            }
        )
    return out
