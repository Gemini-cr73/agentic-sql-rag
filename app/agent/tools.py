# app/agent/tools.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from app.retrieval.service import retrieve_rows_for_api


def _normalize_mode(mode: str) -> str:
    m = (mode or "hybrid").strip().lower()
    if m not in {"hybrid", "fts", "vector"}:
        return "hybrid"
    return m


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


def _to_dict(x: Any) -> dict[str, Any]:
    """
    Coerce a retrieval hit into a plain dict.
    Handles:
      - dict
      - dataclass
      - SQLAlchemy Row / RowMapping
      - generic objects with __dict__
    """
    if x is None:
        return {}

    if isinstance(x, dict):
        raw = x
    elif is_dataclass(x):
        raw = asdict(x)
    else:
        mapping = getattr(x, "_mapping", None)
        if mapping is not None:
            raw = dict(mapping)
        else:
            d = getattr(x, "__dict__", None)
            raw = dict(d) if isinstance(d, dict) else {"value": str(x)}

    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    offsets = raw.get("offsets") if isinstance(raw.get("offsets"), dict) else {}

    return {
        "stable_id": str(raw.get("stable_id"))
        if raw.get("stable_id") is not None
        else None,
        "doc_id": _safe_int(raw.get("doc_id") or raw.get("document_id")),
        "chunk_id": _safe_int(raw.get("chunk_id")),
        "text": str(raw.get("text") or raw.get("content") or ""),
        "char_start": _safe_int(
            raw.get("char_start")
            if raw.get("char_start") is not None
            else offsets.get("char_start")
        ),
        "char_end": _safe_int(
            raw.get("char_end")
            if raw.get("char_end") is not None
            else offsets.get("char_end")
        ),
        "title": raw.get("title") or metadata.get("title"),
        "source": raw.get("source") or metadata.get("source"),
        "fts_score": _safe_float(raw.get("fts_score")),
        "vec_distance": _safe_float(
            raw.get("vec_distance") or raw.get("vector_distance")
        ),
        "vec_similarity": _safe_float(
            raw.get("vec_similarity") or raw.get("vector_score")
        ),
        "hybrid_score": _safe_float(raw.get("hybrid_score")),
        "base_score": _safe_float(raw.get("base_score")),
        "rerank_score": _safe_float(raw.get("rerank_score")),
        "final_score": _safe_float(raw.get("final_score")),
        "rerank_method": raw.get("rerank_method"),
    }


def tool_search(
    *,
    query: str,
    mode: str = "hybrid",
    alpha: float = 0.6,
    k_final: int = 8,
) -> list[dict[str, Any]]:
    """
    Tool wrapper around retrieve_rows_for_api.
    Always returns list[dict] to keep the agent loop simple/consistent.
    """
    q = (query or "").strip()
    if not q:
        return []

    m = _normalize_mode(mode)

    hits = retrieve_rows_for_api(
        query=q,
        mode=m,
        alpha=float(alpha),
        k_final=int(k_final),
    )

    return [_to_dict(h) for h in (hits or []) if h is not None]
