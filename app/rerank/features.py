from __future__ import annotations

from typing import Any

from app.rerank.heuristic import overlap_features


def build_feature_vector(query: str, row: dict[str, Any]) -> list[float]:
    """
    Build ML features for reranking.
    """

    text = str(row.get("text") or "")

    lexical = overlap_features(query, text)

    hybrid_score = float(row.get("hybrid_score") or 0.0)
    fts_score = float(row.get("fts_score") or 0.0)
    vec_score = float(row.get("vec_similarity") or row.get("vector_score") or 0.0)

    features = [
        lexical["matched_terms"],
        lexical["overlap_ratio"],
        lexical["contains_full_query"],
        lexical["query_len"],
        lexical["text_len"],
        hybrid_score,
        fts_score,
        vec_score,
    ]

    return features
