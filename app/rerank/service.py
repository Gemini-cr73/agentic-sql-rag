from __future__ import annotations

from typing import Any

from app.rerank.heuristic import _overlap_score


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _base_score(row: dict[str, Any]) -> float:
    """
    Determine the base retrieval score from a row dict.

    Preference order:
    1) explicit base_score
    2) hybrid_score
    3) fts_score
    4) vec_similarity / vector_score
    5) negative vec_distance (smaller distance is better)
    """
    if row.get("base_score") is not None:
        return _safe_float(row.get("base_score"))

    if row.get("hybrid_score") is not None:
        return _safe_float(row.get("hybrid_score"))

    if row.get("fts_score") is not None:
        return _safe_float(row.get("fts_score"))

    if row.get("vec_similarity") is not None:
        return _safe_float(row.get("vec_similarity"))

    if row.get("vector_score") is not None:
        return _safe_float(row.get("vector_score"))

    if row.get("vec_distance") is not None:
        return -_safe_float(row.get("vec_distance"))

    return 0.0


def _overlap_rerank_score(query: str, row: dict[str, Any]) -> float:
    text = str(row.get("text") or "")
    return _safe_float(_overlap_score(query, text))


def _ml_score(query: str, row: dict[str, Any]) -> float:
    """
    Lazy-load ML reranker so the heuristic path still works even if the ML
    module has not been created yet.
    """
    from app.rerank.ml_reranker import predict_ml_rerank_score

    return _safe_float(predict_ml_rerank_score(query, row))


def _compute_rerank_score(
    method: str, query: str, row: dict[str, Any]
) -> tuple[float, str, str | None]:
    """
    Returns:
      (rerank_score, rerank_method_used, rerank_note)

    Behavior:
    - overlap -> always uses heuristic overlap scoring
    - ml -> tries ML reranker, but gracefully falls back to overlap if the
      ML module/artifacts are not available
    """
    method_norm = (method or "overlap").strip().lower()

    if method_norm == "overlap":
        return _overlap_rerank_score(query, row), "overlap", None

    if method_norm == "ml":
        try:
            return _ml_score(query, row), "ml", None
        except Exception:
            fallback_score = _overlap_rerank_score(query, row)
            return (
                fallback_score,
                "overlap",
                "Requested rerank method 'ml' was unavailable; used 'overlap' fallback.",
            )

    raise ValueError(f"Unknown rerank method: {method}")


def rerank_rows(
    query: str,
    rows: list[dict[str, Any]],
    *,
    enabled: bool = False,
    method: str = "overlap",
    weight: float = 0.15,
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """
    Returns:
      (reranked_rows, pre_ids, post_ids)

    Adds to each reranked row:
      - rerank_method
      - rerank_score
      - base_score
      - final_score

    Optional metadata:
      - rerank_note
    """
    if not rows:
        return [], [], []

    pre_ids = [str(r.get("stable_id")) for r in rows]

    # If rerank is disabled, still ensure the rows have stable score fields
    if not enabled:
        out: list[dict[str, Any]] = []
        for r in rows:
            rr = dict(r)
            base = _base_score(rr)
            rr["base_score"] = base
            rr["rerank_score"] = rr.get("rerank_score")
            rr["final_score"] = (
                _safe_float(rr.get("final_score"))
                if rr.get("final_score") is not None
                else base
            )
            rr["rerank_method"] = rr.get("rerank_method")
            if rr.get("rerank_note") is None:
                rr["rerank_note"] = None
            out.append(rr)

        ids = [str(r.get("stable_id")) for r in out]
        return out, ids, ids

    n = len(rows)
    k = min(max(int(top_k or n), 1), n)

    head = rows[:k]
    tail = rows[k:]

    scored: list[dict[str, Any]] = []
    for r in head:
        rerank_score, rerank_method_used, rerank_note = _compute_rerank_score(
            method, query, r
        )
        base = _base_score(r)
        final = base + float(weight) * rerank_score

        rr = dict(r)
        rr["rerank_method"] = rerank_method_used
        rr["rerank_score"] = rerank_score
        rr["base_score"] = base
        rr["final_score"] = final
        rr["rerank_note"] = rerank_note
        scored.append(rr)

    # Tail rows keep their original order, but still get normalized score fields
    normalized_tail: list[dict[str, Any]] = []
    for r in tail:
        rr = dict(r)
        base = _base_score(rr)
        rr["base_score"] = base
        rr["rerank_score"] = rr.get("rerank_score")
        rr["final_score"] = (
            _safe_float(rr.get("final_score"))
            if rr.get("final_score") is not None
            else base
        )
        rr["rerank_method"] = rr.get("rerank_method")
        if rr.get("rerank_note") is None:
            rr["rerank_note"] = None
        normalized_tail.append(rr)

    scored.sort(
        key=lambda x: (
            _safe_float(x.get("final_score")),
            _safe_float(x.get("rerank_score")),
            str(x.get("stable_id", "")),
        ),
        reverse=True,
    )

    out = scored + normalized_tail
    post_ids = [str(r.get("stable_id")) for r in out]
    return out, pre_ids, post_ids
