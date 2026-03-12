# app/schemas/retrieve.py
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, confloat, conint


class RetrievalMode(str, Enum):
    hybrid = "hybrid"
    fts = "fts"
    vector = "vector"


class MergeStrategy(str, Enum):
    union = "union"
    intersection = "intersection"
    rrf = "rrf"


class RetrieveFilters(BaseModel):
    source: str | None = None
    title: str | None = None
    doc_id: int | None = None
    tags_any: list[str] | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: RetrievalMode = RetrievalMode.hybrid
    alpha: confloat(ge=0.0, le=1.0) = 0.6
    k_final: conint(ge=1, le=100) = 10
    merge_strategy: MergeStrategy = MergeStrategy.union
    filters: RetrieveFilters | None = None


class RetrievedRow(BaseModel):
    stable_id: str | None = None
    doc_id: int | None = None
    chunk_id: int | None = None
    text: str = ""
    char_start: int | None = None
    char_end: int | None = None
    title: str | None = None
    source: str | None = None

    fts_score: float | None = None
    vec_distance: float | None = None
    vec_similarity: float | None = None
    hybrid_score: float | None = None

    base_score: float | None = None
    rerank_score: float | None = None
    final_score: float | None = None
    rerank_method: str | None = None


class RetrieveResponse(BaseModel):
    query: str
    mode: RetrievalMode
    alpha: float
    k_final: int
    merge_strategy: MergeStrategy
    results: list[RetrievedRow]
