from __future__ import annotations

import argparse
import hashlib
import os

from sqlalchemy import text

from app.db.session import new_session

DIM = int(os.getenv("EMBED_DIM", "384"))


def _fake_embedding(text_value: str, dim: int = DIM) -> list[float]:
    """
    Deterministic embedding stub with no numpy dependency.
    Produces the same vector for the same text every time.
    """
    digest = hashlib.sha256(text_value.encode("utf-8", errors="ignore")).digest()
    out: list[float] = []

    for i in range(dim):
        b = digest[i % len(digest)]
        out.append((b / 255.0) * 2.0 - 1.0)  # map to [-1, 1]

    return out


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


def backfill_embeddings(limit: int = 500) -> int:
    with new_session() as s:
        rows = s.execute(
            text(
                """
                SELECT id, text
                FROM doc_chunks
                WHERE embedding IS NULL
                ORDER BY id
                LIMIT :limit
                """
            ),
            {"limit": int(limit)},
        ).fetchall()

        if not rows:
            print("No missing embeddings to backfill.")
            return 0

        updated = 0
        for chunk_id, chunk_text in rows:
            emb = _fake_embedding(chunk_text or "")
            emb_literal = _vec_literal(emb)

            s.execute(
                text(
                    """
                    UPDATE doc_chunks
                    SET embedding = CAST(:emb AS vector)
                    WHERE id = :id
                    """
                ),
                {
                    "emb": emb_literal,
                    "id": int(chunk_id),
                },
            )
            updated += 1

        s.commit()
        print(f"Backfilled embeddings for {updated} chunks.")
        return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    backfill_embeddings(limit=args.limit)


if __name__ == "__main__":
    main()
