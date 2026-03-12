from __future__ import annotations

import json
import traceback
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.loop import run_agent

# Keep the prefix here so the route is POST /agent/ask
router = APIRouter(prefix="/agent", tags=["agent"])


class AgentAskRequest(BaseModel):
    # Upgrade 1 frontend currently sends "query"
    query: str = Field(..., min_length=1)
    mode: str = "hybrid"
    alpha: float = 0.6
    k_final: int = 5

    # Keep these compatible with current / future agent loop usage
    use_agent_memory: bool = False
    max_iterations: int = 3


class AgentToolCall(BaseModel):
    name: str = "unknown"
    arguments: dict[str, Any] = {}
    result: dict[str, Any] | list[Any] | str | None = None


class AgentCitation(BaseModel):
    id: int | None = None
    stable_id: str | None = None
    doc_id: int | None = None
    chunk_id: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    title: str | None = None
    source: str | None = None


class AgentGroundingReport(BaseModel):
    grounding_score: float = 0.0
    supported_sentences: int = 0
    total_sentences: int = 0
    has_hallucinations: bool = False
    sentence_checks: list[dict[str, Any]] = []


class AgentAskResponse(BaseModel):
    query: str
    answer: str
    citations: list[AgentCitation] = []
    tools: list[AgentToolCall] = []
    grounding: AgentGroundingReport | None = None


def _to_tool_calls(raw_tool_calls: Any) -> list[AgentToolCall]:
    """
    Normalize tool calls into list[AgentToolCall].
    Accepts:
      - None -> []
      - list[AgentToolCall]
      - list[dict]
      - list[objects with model_dump/__dict__]
    """
    if raw_tool_calls is None:
        return []

    if not isinstance(raw_tool_calls, list):
        raise HTTPException(
            status_code=500,
            detail=f"run_agent() tool_calls must be list, got {type(raw_tool_calls).__name__}",
        )

    out: list[AgentToolCall] = []
    for tc in raw_tool_calls:
        if tc is None:
            continue

        if isinstance(tc, AgentToolCall):
            out.append(tc)
            continue

        if isinstance(tc, dict):
            out.append(
                AgentToolCall(
                    name=str(tc.get("name", "unknown")),
                    arguments=tc.get("arguments", {})
                    if isinstance(tc.get("arguments", {}), dict)
                    else {},
                    result=tc.get("result"),
                )
            )
            continue

        if hasattr(tc, "model_dump"):
            dumped = tc.model_dump()
            if isinstance(dumped, dict):
                out.append(
                    AgentToolCall(
                        name=str(dumped.get("name", "unknown")),
                        arguments=dumped.get("arguments", {})
                        if isinstance(dumped.get("arguments", {}), dict)
                        else {},
                        result=dumped.get("result"),
                    )
                )
            continue

        if hasattr(tc, "__dict__"):
            dumped = dict(tc.__dict__)
            out.append(
                AgentToolCall(
                    name=str(dumped.get("name", "unknown")),
                    arguments=dumped.get("arguments", {})
                    if isinstance(dumped.get("arguments", {}), dict)
                    else {},
                    result=dumped.get("result"),
                )
            )
            continue

        raise HTTPException(
            status_code=500,
            detail=f"Each tool_call must be dict-like, got {type(tc).__name__}",
        )

    return out


def _coerce_citations(raw: Any) -> list[AgentCitation]:
    """
    Normalize citations into frontend-safe citation objects.
    Accepts:
      - None -> []
      - dict -> [dict]
      - list[dict]
      - JSON string of dict/list
      - pydantic objects / __dict__ objects
    """
    if raw is None:
        return []

    if isinstance(raw, str):
        s = raw.strip()
        try:
            raw = json.loads(s)
        except Exception:
            return []

    if isinstance(raw, dict):
        raw = [raw]

    if not isinstance(raw, list):
        raise HTTPException(
            status_code=500,
            detail=f"run_agent() citations must be list|dict|json-str, got {type(raw).__name__}",
        )

    out: list[AgentCitation] = []

    for c in raw:
        if c is None:
            continue

        if isinstance(c, dict):
            out.append(
                AgentCitation(
                    id=c.get("id"),
                    stable_id=str(c.get("stable_id"))
                    if c.get("stable_id") is not None
                    else None,
                    doc_id=c.get("doc_id"),
                    chunk_id=c.get("chunk_id"),
                    char_start=c.get("char_start"),
                    char_end=c.get("char_end"),
                    title=c.get("title"),
                    source=c.get("source"),
                )
            )
            continue

        if isinstance(c, str):
            try:
                parsed = json.loads(c)
                if isinstance(parsed, dict):
                    out.append(
                        AgentCitation(
                            id=parsed.get("id"),
                            stable_id=str(parsed.get("stable_id"))
                            if parsed.get("stable_id") is not None
                            else None,
                            doc_id=parsed.get("doc_id"),
                            chunk_id=parsed.get("chunk_id"),
                            char_start=parsed.get("char_start"),
                            char_end=parsed.get("char_end"),
                            title=parsed.get("title"),
                            source=parsed.get("source"),
                        )
                    )
                elif isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            out.append(
                                AgentCitation(
                                    id=item.get("id"),
                                    stable_id=str(item.get("stable_id"))
                                    if item.get("stable_id") is not None
                                    else None,
                                    doc_id=item.get("doc_id"),
                                    chunk_id=item.get("chunk_id"),
                                    char_start=item.get("char_start"),
                                    char_end=item.get("char_end"),
                                    title=item.get("title"),
                                    source=item.get("source"),
                                )
                            )
            except Exception:
                continue
            continue

        if hasattr(c, "model_dump"):
            dumped = c.model_dump()
            if isinstance(dumped, dict):
                out.append(
                    AgentCitation(
                        id=dumped.get("id"),
                        stable_id=str(dumped.get("stable_id"))
                        if dumped.get("stable_id") is not None
                        else None,
                        doc_id=dumped.get("doc_id"),
                        chunk_id=dumped.get("chunk_id"),
                        char_start=dumped.get("char_start"),
                        char_end=dumped.get("char_end"),
                        title=dumped.get("title"),
                        source=dumped.get("source"),
                    )
                )
            continue

        if hasattr(c, "__dict__"):
            dumped = dict(c.__dict__)
            out.append(
                AgentCitation(
                    id=dumped.get("id"),
                    stable_id=str(dumped.get("stable_id"))
                    if dumped.get("stable_id") is not None
                    else None,
                    doc_id=dumped.get("doc_id"),
                    chunk_id=dumped.get("chunk_id"),
                    char_start=dumped.get("char_start"),
                    char_end=dumped.get("char_end"),
                    title=dumped.get("title"),
                    source=dumped.get("source"),
                )
            )
            continue

    return out


def _coerce_grounding(raw: Any) -> AgentGroundingReport | None:
    """
    Agent mode may or may not return grounding yet.
    Keep it optional for Upgrade 1.
    """
    if raw is None:
        return None

    if isinstance(raw, AgentGroundingReport):
        return raw

    if isinstance(raw, dict):
        return AgentGroundingReport(
            grounding_score=float(raw.get("grounding_score", 0.0)),
            supported_sentences=int(raw.get("supported_sentences", 0)),
            total_sentences=int(raw.get("total_sentences", 0)),
            has_hallucinations=bool(raw.get("has_hallucinations", False)),
            sentence_checks=raw.get("sentence_checks", [])
            if isinstance(raw.get("sentence_checks", []), list)
            else [],
        )

    return None


@router.post("/ask", response_model=AgentAskResponse)
def ask(payload: AgentAskRequest) -> AgentAskResponse:
    try:
        out = run_agent(
            question=payload.query,
            mode=payload.mode,
            alpha=float(payload.alpha),
            k_final=int(payload.k_final),
            max_steps=int(payload.max_iterations),
        )
    except Exception as e:
        detail = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=detail)

    if not isinstance(out, dict):
        raise HTTPException(
            status_code=500,
            detail=f"run_agent() must return dict, got {type(out).__name__}",
        )

    tool_calls = _to_tool_calls(out.get("tool_calls") or out.get("tools"))
    citations = _coerce_citations(out.get("citations"))
    grounding = _coerce_grounding(out.get("grounding"))

    return AgentAskResponse(
        query=payload.query,
        answer=str(out.get("answer", "")),
        citations=citations,
        tools=tool_calls,
        grounding=grounding,
    )
