from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text or "")]


def overlap_features(query: str, text: str) -> dict[str, float]:
    """
    Shared lexical features for heuristic reranking and future ML reranking.
    """
    q = _tokens(query)
    t = _tokens(text)

    if not q or not t:
        return {
            "query_unique_terms": 0.0,
            "text_unique_terms": 0.0,
            "matched_terms": 0.0,
            "overlap_ratio": 0.0,
            "contains_full_query": 0.0,
            "query_len": float(len(q)),
            "text_len": float(len(t)),
        }

    q_unique = set(q)
    t_unique = set(t)
    matched_terms = q_unique.intersection(t_unique)

    return {
        "query_unique_terms": float(len(q_unique)),
        "text_unique_terms": float(len(t_unique)),
        "matched_terms": float(len(matched_terms)),
        "overlap_ratio": float(len(matched_terms)) / float(len(q_unique)),
        "contains_full_query": float(" ".join(q) in " ".join(t)),
        "query_len": float(len(q)),
        "text_len": float(len(t)),
    }


def _overlap_score(query: str, text: str) -> float:
    """
    Lightweight "cross-encoder-like" heuristic:
    - tokenize query + text
    - score by weighted token overlap (query term frequency boosts)
    - add a small bonus if the full query phrase appears in the text
    """
    q = _tokens(query)
    t = _tokens(text)

    if not q or not t:
        return 0.0

    q_counts = Counter(q)
    t_set = set(t)

    score = 0.0
    for term, tf in q_counts.items():
        if term in t_set:
            score += 1.0 + 0.25 * min(tf - 1, 3)

    if " ".join(q) in " ".join(t):
        score += 0.5

    return score / float(len(set(q)))
