from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import text

from app.db.session import new_session

try:
    # You should have this in: app/embeddings/stub.py
    # def embed_text(text: str) -> list[float]:
    from app.embeddings.stub import embed_text  # type: ignore
except Exception:  # pragma: no cover
    embed_text = None  # type: ignore


@dataclass(frozen=True)
class RetrievalHit:
    stable_id: str
    document_id: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    score: float


def _to_pgvector_literal(vec: Sequence[float]) -> str:
    # Convert Python list -> pgvector literal: "[0.1,0.2,...]"
    return "[" + ",".join(f"{float(x):.10f}" for x in vec) + "]"


def _coerce_to_vector(query: str | list[float]) -> list[float]:
    """
    If query is already a vector, return it.
    If query is text, embed it using embed_text().
    """
    if isinstance(query, list):
        return query

    if not isinstance(query, str):
        raise TypeError(f"vector_search expected str or list[float], got {type(query)}")

    if embed_text is None:
        raise ImportError(
            "vector_search received query text, but no embed_text() function could be imported. "
            "Fix: create app/embeddings/stub.py with embed_text(text)->list[float], "
            "or update vector.py to import your actual embedder."
        )

    return embed_text(query)


def vector_search(query: str | list[float], k: int = 5) -> list[RetrievalHit]:
    """
    Vector similarity search using pgvector.

    Accepts:
      - query as text (str) -> embedded via embed_text()
      - query as list[float] -> used directly

    IMPORTANT: Cast bind parameter to ::vector so pgvector operators work.
    """
    query_vec = _coerce_to_vector(query)
    qvec = _to_pgvector_literal(query_vec)

    sql = text(
        """
        SELECT
            stable_id,
            document_id,
            chunk_index,
            char_start,
            char_end,
            text,
            (1.0 / (1.0 + (embedding <=> (:qvec)::vector))) AS score
        FROM doc_chunks
        WHERE embedding IS NOT NULL
        ORDER BY (embedding <=> (:qvec)::vector) ASC, stable_id ASC
        LIMIT :k
        """
    )

    db = new_session()
    try:
        rows = db.execute(sql, {"qvec": qvec, "k": k}).fetchall()
        return [
            RetrievalHit(
                stable_id=str(r[0]),
                document_id=int(r[1]),
                chunk_index=int(r[2]),
                char_start=int(r[3]),
                char_end=int(r[4]),
                text=str(r[5]),
                score=float(r[6] or 0.0),
            )
            for r in rows
        ]
    finally:
        db.close()
