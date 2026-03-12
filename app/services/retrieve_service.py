# app/services/retrieve_service.py
from __future__ import annotations

from typing import Any

from app.rerank import rerank_rows
from app.retrieval.service import retrieve_rows_for_api
from app.schemas.retrieve import RetrievalMode, RetrieveFilters


def _normalize_mode(mode: str | RetrievalMode) -> str:
    if isinstance(mode, RetrievalMode):
        mode_str = mode.value
    else:
        mode_str = (mode or "hybrid").lower().strip()

    if mode_str not in {"hybrid", "fts", "vector"}:
        return "hybrid"
    return mode_str


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


def _normalize_filters(
    filters: RetrieveFilters | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if filters is None:
        return None
    if isinstance(filters, RetrieveFilters):
        return filters.model_dump(exclude_none=True)
    if isinstance(filters, dict):
        return {kk: vv for kk, vv in filters.items() if vv is not None}
    return None


def _normalize_row(row: Any) -> dict[str, Any] | None:
    """
    Convert raw retrieval rows into a frontend-safe flat dict so Upgrade 1 UI
    can consistently show citations, evidence cards, and rerank metrics.
    """
    if row is None:
        return None

    if isinstance(row, dict):
        raw = row
    elif hasattr(row, "model_dump"):
        raw = row.model_dump()
    elif hasattr(row, "__dict__"):
        raw = dict(row.__dict__)
    else:
        return None

    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    offsets = raw.get("offsets") if isinstance(raw.get("offsets"), dict) else {}

    stable_id = raw.get("stable_id")
    title = raw.get("title") or metadata.get("title")
    source = raw.get("source") or metadata.get("source")

    normalized: dict[str, Any] = {
        "stable_id": str(stable_id) if stable_id is not None else None,
        "doc_id": raw.get("doc_id") or raw.get("document_id"),
        "chunk_id": raw.get("chunk_id"),
        "title": title,
        "source": source,
        "char_start": raw.get("char_start")
        if raw.get("char_start") is not None
        else offsets.get("char_start"),
        "char_end": raw.get("char_end")
        if raw.get("char_end") is not None
        else offsets.get("char_end"),
        "text": str(raw.get("text") or ""),
        "base_score": raw.get("base_score"),
        "rerank_score": raw.get("rerank_score"),
        "final_score": raw.get("final_score"),
        "hybrid_score": raw.get("hybrid_score"),
        "fts_score": raw.get("fts_score"),
        "vector_score": raw.get("vector_score"),
        "rerank_method": raw.get("rerank_method"),
    }

    return normalized


def retrieve(
    *,
    query: str,
    mode: str | RetrievalMode = RetrievalMode.hybrid,
    alpha: float = 0.6,
    k_final: int = 10,
    filters: RetrieveFilters | dict[str, Any] | None = None,
    merge_strategy: str | None = None,
    rerank: bool = False,
    rerank_method: str = "overlap",
    rerank_weight: float = 0.15,
    rerank_top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieval service used by API layer.

    Responsibilities:
    - normalize mode/alpha/k
    - call core retrieval pipeline
    - optionally rerank rows
    - flatten row metadata into a stable response shape for Upgrade 1 UI
    """

    mode_str = _normalize_mode(mode)
    a = _clamp_alpha(alpha)
    k = _clamp_k(k_final)
    filters_dict = _normalize_filters(filters)

    # Reserved for future SQL filter support / merge handling.
    _ = filters_dict
    _ = merge_strategy

    rows = retrieve_rows_for_api(
        query=query,
        mode=mode_str,
        alpha=a,
        k_final=k,
    )

    if rows is None:
        rows = []

    rows, _pre_ids, _post_ids = rerank_rows(
        query,
        rows,
        enabled=bool(rerank),
        method=str(rerank_method or "overlap"),
        weight=float(rerank_weight),
        top_k=(rerank_top_k if rerank_top_k is not None else k),
    )

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = _normalize_row(row)
        if flat is not None:
            normalized_rows.append(flat)

    return normalized_rows
