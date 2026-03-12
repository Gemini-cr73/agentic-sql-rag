# app/retrieval/service.py
from __future__ import annotations

from typing import Any

from app.retrieval.hybrid import MergeStrategy, hybrid_search


def _normalize_mode(mode: str) -> str:
    mode_norm = (mode or "hybrid").lower().strip()
    if mode_norm not in {"hybrid", "fts", "vector"}:
        return "hybrid"
    return mode_norm


def _clamp_alpha(alpha: float) -> float:
    a = float(alpha)
    if a < 0.0:
        return 0.0
    if a > 1.0:
        return 1.0
    return a


def _clamp_k(k_final: int) -> int:
    k = int(k_final)
    if k < 1:
        return 1
    if k > 100:
        return 100
    return k


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def retrieve_rows_for_api(
    *,
    query: str,
    mode: str = "hybrid",
    alpha: float = 0.6,
    k_final: int = 10,
    merge_strategy: MergeStrategy = "union",
) -> list[dict[str, Any]]:
    """
    API-facing adapter:
      - validates inputs
      - calls hybrid_search()
      - returns a stable flat row shape for API + UI use
    """
    q = (query or "").strip()
    if not q:
        return []

    mode_norm = _normalize_mode(mode)
    a = _clamp_alpha(alpha)
    k = _clamp_k(k_final)

    use_fts = mode_norm in {"hybrid", "fts"}
    use_vector = mode_norm in {"hybrid", "vector"}

    hits = hybrid_search(
        query=q,
        k=k,
        alpha=a,
        use_fts=use_fts,
        use_vector=use_vector,
        merge_strategy=merge_strategy,
    )

    out: list[dict[str, Any]] = []
    for h in hits:
        stable_id = getattr(h, "stable_id", None)
        doc_id = getattr(h, "document_id", None)
        chunk_id = getattr(h, "chunk_id", None)
        text = getattr(h, "content", "") or ""
        char_start = getattr(h, "char_start", None)
        char_end = getattr(h, "char_end", None)
        title = getattr(h, "title", None)
        source = getattr(h, "source", None)

        fts_score = _safe_float(getattr(h, "fts_score", None))
        vec_distance = _safe_float(getattr(h, "vector_distance", None))
        vec_similarity = _safe_float(getattr(h, "vector_score", None))
        hybrid_score = _safe_float(getattr(h, "hybrid_score", None))

        # Base score used later by rerank + UI summary
        if hybrid_score is not None:
            base_score = hybrid_score
        elif fts_score is not None:
            base_score = fts_score
        elif vec_similarity is not None:
            base_score = vec_similarity
        else:
            base_score = 0.0

        out.append(
            {
                "stable_id": str(stable_id) if stable_id is not None else None,
                "doc_id": _safe_int(doc_id),
                "chunk_id": _safe_int(chunk_id),
                "text": str(text),
                "char_start": _safe_int(char_start),
                "char_end": _safe_int(char_end),
                "title": title,
                "source": source,
                "fts_score": fts_score,
                "vec_distance": vec_distance,
                "vec_similarity": vec_similarity,
                "hybrid_score": hybrid_score,
                "base_score": base_score,
                # placeholders preserved for later rerank stage
                "rerank_score": None,
                "final_score": base_score,
                "rerank_method": None,
            }
        )

    return out
