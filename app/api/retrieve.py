# app/api/retrieve.py
from __future__ import annotations

import inspect
import time
import traceback

from fastapi import APIRouter, HTTPException

from app.evaluation.retrieval_logging import (
    RetrievalLogEvent,
    Timer,
    log_retrieval_event,
)
from app.schemas.retrieve import RetrievedRow, RetrieveRequest, RetrieveResponse
from app.services.retrieve_service import retrieve

router = APIRouter(tags=["retrieval"])


@router.post("/retrieve", response_model=RetrieveResponse)
def post_retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    t = Timer()

    try:
        kwargs = {
            "query": payload.query,
            "mode": payload.mode,
            "alpha": float(payload.alpha),
            "k_final": int(payload.k_final),
            "filters": payload.filters,
        }

        sig = inspect.signature(retrieve)
        if "merge_strategy" in sig.parameters:
            kwargs["merge_strategy"] = payload.merge_strategy

        rows = retrieve(**kwargs)
        latency_ms = t.ms()

        results: list[RetrievedRow] = []
        for r in rows:
            results.append(
                RetrievedRow(
                    stable_id=str(r.get("stable_id"))
                    if r.get("stable_id") is not None
                    else None,
                    doc_id=r.get("doc_id"),
                    chunk_id=r.get("chunk_id"),
                    text=str(r.get("text") or ""),
                    char_start=r.get("char_start"),
                    char_end=r.get("char_end"),
                    title=r.get("title"),
                    source=r.get("source"),
                    fts_score=r.get("fts_score"),
                    vec_distance=r.get("vec_distance"),
                    vec_similarity=r.get("vec_similarity"),
                    hybrid_score=r.get("hybrid_score"),
                    base_score=r.get("base_score"),
                    rerank_score=r.get("rerank_score"),
                    final_score=r.get("final_score"),
                    rerank_method=r.get("rerank_method"),
                )
            )

        scores_payload: list[dict[str, object]] = []
        for r in rows:
            scores_payload.append(
                {
                    "stable_id": r.get("stable_id"),
                    "doc_id": r.get("doc_id"),
                    "chunk_id": r.get("chunk_id"),
                    "fts_score": r.get("fts_score"),
                    "vec_similarity": r.get("vec_similarity"),
                    "hybrid_score": r.get("hybrid_score"),
                    "base_score": r.get("base_score"),
                    "rerank_score": r.get("rerank_score"),
                    "final_score": r.get("final_score"),
                    "rerank_method": r.get("rerank_method"),
                }
            )

        try:
            log_retrieval_event(
                RetrievalLogEvent(
                    ts=time.time(),
                    query=payload.query,
                    mode=payload.mode,
                    top_k=int(payload.k_final),
                    latency_ms=latency_ms,
                    scores=scores_payload,
                )
            )
        except Exception:
            pass

        return RetrieveResponse(
            query=payload.query,
            mode=payload.mode,
            alpha=float(payload.alpha),
            k_final=int(payload.k_final),
            merge_strategy=payload.merge_strategy,
            results=results,
        )

    except Exception as e:
        detail = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=detail)
