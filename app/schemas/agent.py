# app/schemas/agent.py
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, confloat, conint


class AgentAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)

    mode: str = Field(default="hybrid", description="hybrid | fts | vector")
    alpha: confloat(ge=0.0, le=1.0) = Field(
        default=0.6, description="Hybrid weight for FTS score"
    )
    k_final: conint(ge=1, le=50) = Field(
        default=8, description="Top-K results to return"
    )

    max_steps: conint(ge=1, le=10) = Field(
        default=4, description="Hard cap on agent steps"
    )

    filters: dict[str, Any] | None = Field(
        default=None, description="Optional future filters"
    )


class AgentToolCall(BaseModel):
    tool: str
    args: dict[str, Any]
    ok: bool
    error: str | None = None


class AgentAskResponse(BaseModel):
    question: str
    answer: str
    tool_calls: list[AgentToolCall]
    citations: list[dict[str, Any]] = Field(default_factory=list)
