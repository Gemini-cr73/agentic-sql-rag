# app/evaluation/metrics.py
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricsAtK:
    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    hits = sum(1 for rid in top if rid in relevant)
    return _safe_div(hits, len(top))  # len(top) can be < k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for rid in top if rid in relevant)
    return _safe_div(hits, len(relevant))


def mrr(retrieved: list[str], relevant: set[str], k: int) -> float:
    # Mean Reciprocal Rank for a single query (0..1)
    top = retrieved[:k]
    for i, rid in enumerate(top, start=1):
        if rid in relevant:
            return 1.0 / float(i)
    return 0.0


def ndcg(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """
    nDCG@K with graded relevance.
    relevant maps doc_id -> relevance (e.g., 0/1/2/3)
    """
    top = retrieved[:k]

    def dcg(items: Iterable[str]) -> float:
        score = 0.0
        for i, rid in enumerate(items, start=1):
            rel = float(relevant.get(rid, 0))
            if rel <= 0:
                continue
            # standard DCG gain: (2^rel - 1) / log2(i+1)
            score += (2.0**rel - 1.0) / math.log2(i + 1.0)
        return score

    actual = dcg(top)

    # Ideal DCG: sort relevant by rel desc
    ideal_order = sorted(relevant.items(), key=lambda kv: kv[1], reverse=True)
    ideal_ids = [doc_id for doc_id, rel in ideal_order][:k]
    ideal = dcg(ideal_ids)

    return _safe_div(actual, ideal)


def compute_metrics_at_k(
    retrieved_ids: list[str],
    relevant_binary: set[str],
    relevant_graded: dict[str, int],
    k: int,
) -> MetricsAtK:
    return MetricsAtK(
        k=k,
        precision_at_k=precision_at_k(retrieved_ids, relevant_binary, k),
        recall_at_k=recall_at_k(retrieved_ids, relevant_binary, k),
        mrr=mrr(retrieved_ids, relevant_binary, k),
        ndcg=ndcg(retrieved_ids, relevant_graded, k),
    )
