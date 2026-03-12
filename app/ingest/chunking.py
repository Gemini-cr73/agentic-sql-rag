from __future__ import annotations

from typing import Any


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    """
    Deterministic character-based chunker.

    Returns a list of dicts:
      - chunk_index
      - char_start
      - char_end
      - text
    """
    text = (text or "").strip()
    if not text:
        return []

    if chunk_size <= 0:
        chunk_size = 1200
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: list[dict[str, Any]] = []
    n = len(text)
    start = 0
    idx = 0
    step = max(1, chunk_size - overlap)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_value = text[start:end]

        chunks.append(
            {
                "chunk_index": idx,
                "char_start": start,
                "char_end": end,
                "text": chunk_value,
            }
        )

        idx += 1
        if end >= n:
            break
        start += step

    return chunks
