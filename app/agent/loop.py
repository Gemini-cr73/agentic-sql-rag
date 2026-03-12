# app/agent/loop.py
from __future__ import annotations

from typing import Any

from app.agent.tools import tool_search
from app.generation.answer_generator import generate_answer
from app.verification.grounding import verify_grounding


def _pick_mode(question: str, requested_mode: str) -> str:
    """
    Simple heuristic:
    - if user explicitly asked for vector/fts/hybrid, respect it
    - otherwise default to hybrid
    """
    rm = (requested_mode or "hybrid").strip().lower()
    if rm in {"hybrid", "fts", "vector"}:
        return rm

    q = (question or "").lower()
    if "vector" in q or "semantic" in q:
        return "vector"
    if "sql" in q or "keyword" in q or "fts" in q:
        return "fts"
    return "hybrid"


def _tool_result_preview(
    rows: list[dict[str, Any]], max_items: int = 3
) -> dict[str, Any]:
    preview: list[dict[str, Any]] = []

    for row in rows[:max_items]:
        preview.append(
            {
                "stable_id": row.get("stable_id"),
                "title": row.get("title"),
                "source": row.get("source"),
                "final_score": row.get("final_score"),
                "rerank_score": row.get("rerank_score"),
            }
        )

    return {
        "count": len(rows),
        "top_results": preview,
    }


def run_agent(
    *,
    question: str,
    mode: str = "hybrid",
    alpha: float = 0.6,
    k_final: int = 8,
    max_steps: int = 4,
) -> dict[str, Any]:
    """
    Minimal bounded agent loop for Upgrade 1.

    Step 1: retrieve evidence via tool_search
    Step 2: generate grounded extractive answer
    Step 3: verify grounding
    """

    chosen_mode = _pick_mode(question, mode)
    if max_steps < 1:
        max_steps = 1

    tools: list[dict[str, Any]] = []

    # Step 1: retrieval
    try:
        rows = tool_search(
            query=question,
            mode=chosen_mode,
            alpha=float(alpha),
            k_final=int(k_final),
        )
        tools.append(
            {
                "name": "search",
                "arguments": {
                    "query": question,
                    "mode": chosen_mode,
                    "alpha": float(alpha),
                    "k_final": int(k_final),
                },
                "result": _tool_result_preview(rows),
            }
        )
    except Exception as e:
        tools.append(
            {
                "name": "search",
                "arguments": {
                    "query": question,
                    "mode": chosen_mode,
                    "alpha": float(alpha),
                    "k_final": int(k_final),
                },
                "result": {"error": str(e)},
            }
        )
        return {
            "answer": "Agent failed during retrieval. See tool trace for details.",
            "tools": tools,
            "citations": [],
            "grounding": {
                "grounding_score": 0.0,
                "supported_sentences": 0,
                "total_sentences": 0,
                "has_hallucinations": False,
                "sentence_checks": [],
            },
        }

    if not rows:
        return {
            "answer": "I couldn’t find any matching evidence in the database for that question.",
            "tools": tools,
            "citations": [],
            "grounding": {
                "grounding_score": 0.0,
                "supported_sentences": 0,
                "total_sentences": 0,
                "has_hallucinations": False,
                "sentence_checks": [],
            },
        }

    # Step 2: answer generation
    answer_obj = generate_answer(
        query=question,
        rows=rows,
        max_sentences=3,
    )

    # Step 3: grounding
    grounding = verify_grounding(
        answer=answer_obj.get("answer", ""),
        citations=answer_obj.get("citations", []),
        rows=rows,
        support_threshold=0.35,
    )

    return {
        "answer": answer_obj.get("answer", ""),
        "tools": tools,
        "tool_calls": tools,  # keep backward compatibility
        "citations": answer_obj.get("citations", []),
        "grounding": grounding,
    }
