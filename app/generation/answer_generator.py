from __future__ import annotations

import re
from typing import Any

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _row_score(row: dict[str, Any]) -> float:
    for key in (
        "final_score",
        "base_score",
        "hybrid_score",
        "fts_score",
        "vector_score",
        "rerank_score",
    ):
        if row.get(key) is not None:
            try:
                return float(row.get(key) or 0.0)
            except Exception:
                pass
    return 0.0


def _clean_sentence(sentence: str) -> str:
    sentence = _WHITESPACE_RE.sub(" ", (sentence or "").strip())
    if not sentence:
        return ""

    # Remove obvious bullet prefixes
    sentence = re.sub(r"^[\-\*\u2022]+\s*", "", sentence).strip()

    # Ensure sentence ends cleanly
    if sentence and sentence[-1] not in ".!?":
        sentence += "."

    return sentence


def _normalize_for_dedup(text: str) -> str:
    return " ".join(_tokens(text))


def _query_overlap_score(query: str, sentence: str) -> float:
    q_tokens = set(_tokens(query))
    s_tokens = set(_tokens(sentence))

    if not q_tokens or not s_tokens:
        return 0.0

    overlap_count = len(q_tokens.intersection(s_tokens))
    return overlap_count / max(len(q_tokens), 1)


def _sentence_score(query: str, sentence: str, row: dict[str, Any]) -> float:
    overlap = _query_overlap_score(query, sentence)
    base = _row_score(row)

    # Weight overlap more heavily so answers feel more on-topic,
    # while still respecting retrieval/rerank score.
    return (2.0 * overlap) + base


def _build_citation(row: dict[str, Any], citation_number: int) -> dict[str, Any]:
    return {
        "id": citation_number,
        "stable_id": row.get("stable_id"),
        "doc_id": row.get("doc_id"),
        "chunk_id": row.get("chunk_id"),
        "char_start": row.get("char_start"),
        "char_end": row.get("char_end"),
        "title": row.get("title"),
        "source": row.get("source"),
    }


def _sentence_is_too_short(sentence: str) -> bool:
    return len(_tokens(sentence)) < 4


def _prefer_diverse_candidates(
    candidates: list[dict[str, Any]],
    max_sentences: int,
) -> list[dict[str, Any]]:
    """
    Select strong candidates while preferring chunk diversity.
    """
    selected: list[dict[str, Any]] = []
    seen_sentences: set[str] = set()
    used_chunk_ids: set[str] = set()

    target = max(1, int(max_sentences))

    # Pass 1: prefer unique chunks
    for item in candidates:
        sentence_key = _normalize_for_dedup(item["sentence"])
        stable_id = str(item["row"].get("stable_id") or "")

        if not sentence_key or sentence_key in seen_sentences:
            continue
        if stable_id and stable_id in used_chunk_ids:
            continue

        selected.append(item)
        seen_sentences.add(sentence_key)
        if stable_id:
            used_chunk_ids.add(stable_id)

        if len(selected) >= target:
            return selected

    # Pass 2: fill remaining slots regardless of chunk reuse
    for item in candidates:
        sentence_key = _normalize_for_dedup(item["sentence"])
        if not sentence_key or sentence_key in seen_sentences:
            continue

        selected.append(item)
        seen_sentences.add(sentence_key)

        if len(selected) >= target:
            break

    return selected


def generate_answer(
    *,
    query: str,
    rows: list[dict[str, Any]],
    max_sentences: int = 3,
) -> dict[str, Any]:
    """
    Improved extractive grounded answer generator.

    Strategy:
    1. score sentences from retrieved rows using query overlap + retrieval score
    2. filter short / noisy sentences
    3. prefer diverse chunks to avoid repetitive answers
    4. append inline citations [1], [2], ...
    """

    if not rows:
        return {
            "answer": "No relevant information was found for this query.",
            "citations": [],
            "used_chunks": [],
        }

    candidates: list[dict[str, Any]] = []

    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue

        sentences = _split_sentences(text)
        if not sentences:
            sentences = [text]

        for sentence in sentences:
            cleaned = _clean_sentence(sentence)
            if not cleaned:
                continue
            if _sentence_is_too_short(cleaned):
                continue

            candidates.append(
                {
                    "sentence": cleaned,
                    "score": _sentence_score(query, cleaned, row),
                    "row": row,
                }
            )

    if not candidates:
        return {
            "answer": "Relevant chunks were retrieved, but no answerable text could be extracted.",
            "citations": [],
            "used_chunks": [],
        }

    candidates.sort(
        key=lambda x: float(x["score"]),
        reverse=True,
    )

    selected = _prefer_diverse_candidates(candidates, max_sentences=max_sentences)

    if not selected:
        return {
            "answer": "Relevant chunks were retrieved, but no answerable text could be extracted.",
            "citations": [],
            "used_chunks": [],
        }

    citations: list[dict[str, Any]] = []
    answer_parts: list[str] = []
    used_chunks: list[str] = []
    chunk_seen: set[str] = set()

    for i, item in enumerate(selected, start=1):
        row = item["row"]
        citation = _build_citation(row, i)
        citations.append(citation)

        stable_id = (
            str(row.get("stable_id")) if row.get("stable_id") is not None else None
        )
        if stable_id and stable_id not in chunk_seen:
            used_chunks.append(stable_id)
            chunk_seen.add(stable_id)

        answer_parts.append(f"{item['sentence']} [{i}]")

    answer_text = " ".join(answer_parts).strip()

    if not answer_text:
        answer_text = (
            "Relevant chunks were retrieved, but no answerable text could be extracted."
        )

    return {
        "answer": answer_text,
        "citations": citations,
        "used_chunks": used_chunks,
    }
