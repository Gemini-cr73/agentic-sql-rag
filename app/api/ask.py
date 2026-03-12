from __future__ import annotations

import traceback
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.generation.answer_generator import generate_answer
from app.services.retrieve_service import retrieve
from app.verification.grounding import verify_grounding

router = APIRouter(tags=["generation"])


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = "hybrid"
    alpha: float = 0.6
    k_final: int = 5

    rerank: bool = True
    rerank_method: str = "ml"
    rerank_weight: float = 0.15
    rerank_top_k: int | None = 5

    max_sentences: int = 3
    support_threshold: float = 0.35


class AskCitation(BaseModel):
    id: int
    stable_id: str | None = None
    doc_id: int | None = None
    chunk_id: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    title: str | None = None
    source: str | None = None


class GroundingSentenceCheck(BaseModel):
    sentence: str
    cited_ids: list[int]
    best_support_score: float
    supported: bool
    best_support_stable_id: str | None = None


class GroundingReport(BaseModel):
    grounding_score: float
    supported_sentences: int
    total_sentences: int
    has_hallucinations: bool
    sentence_checks: list[GroundingSentenceCheck]


class RetrievedRow(BaseModel):
    stable_id: str | None = None
    doc_id: int | None = None
    chunk_id: int | None = None
    title: str | None = None
    source: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    text: str | None = None

    base_score: float | None = None
    rerank_score: float | None = None
    final_score: float | None = None
    rerank_method: str | None = None
    rerank_note: str | None = None


class AskResponse(BaseModel):
    query: str
    mode: str
    answer: str
    citations: list[AskCitation]
    used_chunks: list[str]
    grounding: GroundingReport
    retrieved: list[RetrievedRow] = []


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_retrieved_rows(rows: list[Any]) -> list[RetrievedRow]:
    """
    Convert whatever retrieve() returns into a frontend-safe list of rows
    so the UI can show evidence cards + rerank summary without crashing.
    """
    out: list[RetrievedRow] = []

    for row in rows:
        if row is None:
            continue

        if isinstance(row, dict):
            raw = row
        elif hasattr(row, "model_dump"):
            raw = row.model_dump()
        elif hasattr(row, "__dict__"):
            raw = dict(row.__dict__)
        else:
            continue

        out.append(
            RetrievedRow(
                stable_id=str(raw.get("stable_id"))
                if raw.get("stable_id") is not None
                else None,
                doc_id=raw.get("doc_id"),
                chunk_id=raw.get("chunk_id"),
                title=raw.get("title"),
                source=raw.get("source"),
                char_start=raw.get("char_start"),
                char_end=raw.get("char_end"),
                text=raw.get("text"),
                base_score=_safe_float(raw.get("base_score")),
                rerank_score=_safe_float(raw.get("rerank_score")),
                final_score=_safe_float(raw.get("final_score")),
                rerank_method=raw.get("rerank_method"),
                rerank_note=raw.get("rerank_note"),
            )
        )

    return out


def _empty_grounding_report() -> GroundingReport:
    return GroundingReport(
        grounding_score=0.0,
        supported_sentences=0,
        total_sentences=0,
        has_hallucinations=False,
        sentence_checks=[],
    )


def _fallback_no_results_answer(query: str, mode: str) -> AskResponse:
    return AskResponse(
        query=query,
        mode=mode,
        answer="No relevant information was found for this query.",
        citations=[],
        used_chunks=[],
        grounding=_empty_grounding_report(),
        retrieved=[],
    )


@router.post("/ask", response_model=AskResponse)
def post_ask(payload: AskRequest) -> AskResponse:
    try:
        rows = retrieve(
            query=payload.query,
            mode=payload.mode,
            alpha=float(payload.alpha),
            k_final=int(payload.k_final),
            rerank=bool(payload.rerank),
            rerank_method=str(payload.rerank_method),
            rerank_weight=float(payload.rerank_weight),
            rerank_top_k=payload.rerank_top_k,
        )

        if not rows:
            return _fallback_no_results_answer(payload.query, payload.mode)

        answer_obj = generate_answer(
            query=payload.query,
            rows=rows,
            max_sentences=int(payload.max_sentences),
        )

        raw_citations = answer_obj.get("citations", [])
        citations: list[AskCitation] = []
        for c in raw_citations:
            try:
                citations.append(AskCitation(**c))
            except Exception:
                # Skip malformed citation rows instead of crashing the endpoint
                continue

        grounding_obj = verify_grounding(
            answer=answer_obj.get("answer", ""),
            citations=[c.model_dump() for c in citations],
            rows=rows,
            support_threshold=float(payload.support_threshold),
        )

        retrieved_rows = _normalize_retrieved_rows(rows)

        answer_text = answer_obj.get("answer", "") or ""
        used_chunks = answer_obj.get("used_chunks", []) or []

        if not answer_text.strip():
            answer_text = "No relevant information was found for this query."

        return AskResponse(
            query=payload.query,
            mode=payload.mode,
            answer=answer_text,
            citations=citations,
            used_chunks=used_chunks,
            grounding=GroundingReport(**grounding_obj),
            retrieved=retrieved_rows,
        )

    except Exception as e:
        detail = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=detail) from e
