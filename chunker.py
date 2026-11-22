"""Text chunking utilities for ingestion."""

from typing import List


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks of roughly ``chunk_size`` characters.

    Uses a simple character-based sliding window with word-boundary snapping so
    chunks do not cut words in half. Good enough for a baseline RAG pipeline;
    production systems often use token-based splitters (e.g. RecursiveCharacterTextSplitter).
    """
    text = (text or "").strip()
    if not text:
        return []

    if chunk_size <= 0:
        return [text]

    overlap = max(0, min(chunk_overlap, chunk_size - 1))
    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        # Snap end to a word boundary if possible.
        if end < n:
            snap = text.rfind(" ", start, end)
            if snap != -1 and snap > start + chunk_size // 2:
                end = snap
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks
