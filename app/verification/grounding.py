from __future__ import annotations

import re
from typing import Any

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "")}


def _split_answer_sentences(answer: str) -> list[str]:
    answer = (answer or "").strip()
    if not answer:
        return []
    parts = _SENTENCE_RE.split(answer)
    return [p.strip() for p in parts if p.strip()]


def _strip_citations(sentence: str) -> str:
    return _CITATION_RE.sub("", sentence).strip()


def _extract_citation_ids(sentence: str) -> list[int]:
    return [int(x) for x in _CITATION_RE.findall(sentence)]


def _support_score(sentence: str, chunk_text: str) -> float:
    """
    Support score for one answer sentence against one chunk.

    Heuristics:
    - exact normalized containment => 1.0
    - otherwise token overlap ratio
    """
    sent_clean = _strip_citations(sentence)
    sent_norm = _normalize(sent_clean)
    chunk_norm = _normalize(chunk_text)

    if not sent_norm or not chunk_norm:
        return 0.0

    if sent_norm in chunk_norm:
        return 1.0

    s_tokens = _tokens(sent_clean)
    c_tokens = _tokens(chunk_text)

    if not s_tokens or not c_tokens:
        return 0.0

    overlap = len(s_tokens.intersection(c_tokens)) / max(len(s_tokens), 1)
    return float(overlap)


def verify_grounding(
    *,
    answer: str,
    citations: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    support_threshold: float = 0.35,
) -> dict[str, Any]:
    """
    Verify whether each answer sentence is grounded in retrieved chunks.
    """
    sentences = _split_answer_sentences(answer)

    citation_map: dict[int, dict[str, Any]] = {
        int(c["id"]): c for c in citations if c.get("id") is not None
    }
    row_map: dict[str, dict[str, Any]] = {
        str(r.get("stable_id")): r for r in rows if r.get("stable_id") is not None
    }

    checks: list[dict[str, Any]] = []

    for sentence in sentences:
        clean_sentence = _strip_citations(sentence)
        cited_ids = _extract_citation_ids(sentence)
        cited_rows: list[dict[str, Any]] = []

        for cid in cited_ids:
            citation = citation_map.get(cid)
            if not citation:
                continue

            stable_id = citation.get("stable_id")
            if stable_id is None:
                continue

            row = row_map.get(str(stable_id))
            if row:
                cited_rows.append(row)

        candidate_rows = cited_rows if cited_rows else rows

        best_score = 0.0
        best_stable_id: str | None = None

        for row in candidate_rows:
            chunk_text = str(row.get("text") or "")
            if not chunk_text.strip():
                continue

            score = _support_score(sentence, chunk_text)
            if score > best_score:
                best_score = score
                best_stable_id = (
                    str(row.get("stable_id"))
                    if row.get("stable_id") is not None
                    else None
                )

        supported = best_score >= float(support_threshold)

        checks.append(
            {
                "sentence": clean_sentence,
                "cited_ids": cited_ids,
                "best_support_score": round(best_score, 4),
                "supported": supported,
                "best_support_stable_id": best_stable_id,
            }
        )

    total = len(checks)
    supported_count = sum(1 for c in checks if c["supported"])
    grounding_score = (supported_count / total) if total else 0.0

    return {
        "grounding_score": round(grounding_score, 4),
        "supported_sentences": supported_count,
        "total_sentences": total,
        "has_hallucinations": supported_count < total,
        "sentence_checks": checks,
    }
