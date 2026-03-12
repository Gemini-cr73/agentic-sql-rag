# tests/test_retrieval_core.py
from __future__ import annotations

from app.retrieval.hybrid import fts_search, hybrid_search, vector_search
from app.retrieval.service import retrieve_rows_for_api


def test_fts_search_returns_results(seeded_doc_and_chunk):
    rows = fts_search("retrieval", k=5)
    assert isinstance(rows, list)
    assert len(rows) >= 1


def test_vector_search_returns_results(seeded_doc_and_chunk):
    rows = vector_search("retrieval", k=5)
    assert isinstance(rows, list)
    assert len(rows) >= 1


def test_hybrid_search_is_deterministic(seeded_doc_and_chunk):
    a = hybrid_search("retrieval", k=5, alpha=0.6, use_fts=True, use_vector=True)
    b = hybrid_search("retrieval", k=5, alpha=0.6, use_fts=True, use_vector=True)
    assert [h.stable_id for h in a] == [h.stable_id for h in b]


def test_service_layer_contract(seeded_doc_and_chunk):
    rows = retrieve_rows_for_api(query="retrieval", mode="hybrid", alpha=0.6, k_final=5)
    assert len(rows) >= 1
    r0 = rows[0]
    assert "stable_id" in r0
    assert "doc_id" in r0
    assert "chunk_id" in r0
    assert "text" in r0
    assert "hybrid_score" in r0
