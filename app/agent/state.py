from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ToolName = Literal["search_sql", "search_vector", "search_hybrid"]


@dataclass
class AgentLimits:
    max_steps: int = 6
    max_tool_calls: int = 3
    max_context_chars: int = 6000


@dataclass
class AgentState:
    question: str
    limits: AgentLimits = field(default_factory=AgentLimits)

    steps: int = 0
    tool_calls: int = 0

    # most recent retrieved rows
    results: list[dict[str, Any]] = field(default_factory=list)

    # debug trace
    trace: list[dict[str, Any]] = field(default_factory=list)

    # final answer output
    answer: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
