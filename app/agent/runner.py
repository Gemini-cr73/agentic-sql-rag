from __future__ import annotations

from typing import Any

from app.agent import tools
from app.agent.state import AgentState, ToolName


def _truncate_context(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 50] + "\n...[truncated]..."


def decide_tool(state: AgentState) -> ToolName:
    """
    Deterministic baseline policy (no LLM):
    - short/keyword queries -> FTS
    - longer/natural language -> hybrid
    - if explicitly says "semantic" -> vector
    """
    q = state.question.lower().strip()
    if "semantic" in q or "meaning" in q:
        return "search_vector"
    if len(q.split()) <= 3:
        return "search_sql"
    return "search_hybrid"


def run_agent(state: AgentState, k: int = 5, alpha: float = 0.6) -> dict[str, Any]:
    while state.steps < state.limits.max_steps:
        state.steps += 1

        # stop if we've already produced an answer
        if state.answer is not None:
            break

        if state.tool_calls >= state.limits.max_tool_calls:
            state.trace.append({"event": "stop", "reason": "max_tool_calls"})
            break

        tool_name = decide_tool(state)
        state.tool_calls += 1
        state.trace.append({"event": "tool_call", "tool": tool_name})

        if tool_name == "search_sql":
            rows = tools.search_sql(state.question, k=k)
        elif tool_name == "search_vector":
            rows = tools.search_vector(state.question, k=k)
        else:
            rows = tools.search_hybrid(state.question, k=k, alpha=alpha)

        state.results = rows
        state.trace.append({"event": "tool_result", "n": len(rows)})

        # Synthesize answer (baseline: extract top chunks + citations)
        if not rows:
            state.answer = "No matching evidence found for that query in the current indexed documents."
            state.citations = []
            break

        # Build a simple evidence-first response
        bullets = []
        cites = []
        for r in rows[:k]:
            text = _truncate_context(str(r.get("text", "")), 500)
            bullets.append(f"- {text}")
            cites.append(
                {
                    "stable_id": r.get("stable_id"),
                    "doc_id": r.get("doc_id"),
                    "chunk_id": r.get("chunk_id"),
                    "char_start": r.get("char_start"),
                    "char_end": r.get("char_end"),
                    "hybrid_score": r.get("hybrid_score"),
                }
            )

        state.answer = "Top evidence:\n" + "\n".join(bullets)
        state.citations = cites
        break

    if state.answer is None:
        state.answer = "Stopped before producing an answer (limit reached)."
        state.trace.append({"event": "stop", "reason": "max_steps"})

    return {
        "question": state.question,
        "answer": state.answer,
        "citations": state.citations,
        "trace": state.trace,
        "steps": state.steps,
        "tool_calls": state.tool_calls,
    }
