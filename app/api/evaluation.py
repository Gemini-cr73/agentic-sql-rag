from __future__ import annotations

import traceback

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.evaluation_service import get_evaluation_runs, get_evaluation_summary

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class EvaluationSummaryResponse(BaseModel):
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    average_grounding_score: float = 0.0
    rerank_comparison: dict = Field(default_factory=dict)
    query_count: int = 0
    source: str = ""
    available: bool = False


class EvaluationRunItem(BaseModel):
    run_id: int | str
    query: str = ""
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    grounding_score: float = 0.0
    rerank_enabled: bool = False
    latency_ms: float = 0.0
    notes: str = ""


class EvaluationRunsResponse(BaseModel):
    runs: list[EvaluationRunItem] = Field(default_factory=list)
    total: int = 0
    source: str = ""
    available: bool = False


@router.get("/summary", response_model=EvaluationSummaryResponse)
def evaluation_summary() -> EvaluationSummaryResponse:
    try:
        data = get_evaluation_summary()
        return EvaluationSummaryResponse(**data)
    except Exception as e:
        detail = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=detail) from e


@router.get("/runs", response_model=EvaluationRunsResponse)
def evaluation_runs() -> EvaluationRunsResponse:
    try:
        data = get_evaluation_runs()
        return EvaluationRunsResponse(**data)
    except Exception as e:
        detail = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=detail) from e
