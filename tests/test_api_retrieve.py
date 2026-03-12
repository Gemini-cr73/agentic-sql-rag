# tests/test_api_retrieve.py
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_api_retrieve_ok(seeded_doc_and_chunk):
    client = TestClient(app)

    payload = {
        "query": "retrieval",
        "mode": "hybrid",
        "alpha": 0.6,
        "k_final": 5,
        "filters": None,
    }

    resp = client.post("/retrieval/retrieve", json=payload)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["mode"] == "hybrid"
    assert len(data["results"]) >= 1
