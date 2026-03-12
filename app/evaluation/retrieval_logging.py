# app/evaluation/retrieval_logging.py
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalLogEvent:
    ts: float
    query: str
    mode: str
    top_k: int
    latency_ms: float
    scores: list[dict[str, Any]]


def _default_log_path() -> Path:
    # env override, otherwise logs/retrieval_events.jsonl at repo root
    p = os.getenv("RETRIEVAL_LOG_PATH", "logs/retrieval_events.jsonl")
    return Path(p)


def log_retrieval_event(event: RetrievalLogEvent) -> None:
    path = _default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


class Timer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0
