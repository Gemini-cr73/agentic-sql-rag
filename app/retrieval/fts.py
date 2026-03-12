from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text

from app.db.session import new_session


@dataclass(frozen=True)
class RetrievalHit:
    stable_id: str
    document_id: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    score: float


def fts_search(query: str, k: int = 5) -> list[RetrievalHit]:
    sql = text(
        """
        SELECT
            stable_id,
            document_id,
            chunk_index,
            char_start,
            char_end,
            text,
            ts_rank(tsv, plainto_tsquery('english', :q)) AS score
        FROM doc_chunks
        WHERE tsv @@ plainto_tsquery('english', :q)
        ORDER BY score DESC, stable_id ASC
        LIMIT :k
        """
    )

    db = new_session()
    try:
        rows = db.execute(sql, {"q": query, "k": k}).fetchall()
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
