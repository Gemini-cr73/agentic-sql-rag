from __future__ import annotations

import argparse
from dataclasses import dataclass

from sqlalchemy import text

from app.db.session import SessionLocal


@dataclass
class Hit:
    stable_id: str
    document_id: int
    chunk_index: int
    score: float
    text: str


def search_fts(query: str, k: int) -> list[Hit]:
    sql = text("""
        SELECT
          stable_id,
          document_id,
          chunk_index,
          ts_rank(tsv, plainto_tsquery('english', :q)) AS score,
          text
        FROM doc_chunks
        WHERE tsv @@ plainto_tsquery('english', :q)
        ORDER BY score DESC, stable_id ASC
        LIMIT :k;
    """)

    db = SessionLocal()
    try:
        rows = db.execute(sql, {"q": query, "k": k}).fetchall()
        return [
            Hit(
                stable_id=r[0],
                document_id=r[1],
                chunk_index=r[2],
                score=float(r[3] or 0.0),
                text=r[4],
            )
            for r in rows
        ]
    finally:
        db.close()


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)

    # existing flags
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--no-vector", action="store_true")
    p.add_argument("--no-fts", action="store_true")

    # ✅ Milestone 4 — Optional Reranking Stage
    p.add_argument("--rerank", action="store_true", help="Enable reranking")
    p.add_argument(
        "--rerank-method", default="overlap", help="Rerank method (default: overlap)"
    )
    p.add_argument(
        "--rerank-weight",
        type=float,
        default=0.15,
        help="Rerank weight added to base score",
    )
    p.add_argument(
        "--rerank-top-k",
        type=int,
        default=0,
        help="Rerank only top N candidates (0 = use k)",
    )

    args = p.parse_args()

    print("")
    print(f"Query: {args.query}")
    print(
        f"k={args.k}, alpha={args.alpha}, "
        f"no_fts={args.no_fts}, no_vector={args.no_vector}, "
        f"rerank={args.rerank}, rerank_method={args.rerank_method}, "
        f"rerank_weight={args.rerank_weight}, rerank_top_k={args.rerank_top_k}"
    )
    print("")

    if args.no_fts:
        print("FTS disabled (--no-fts). Nothing to run yet.")
        return

    hits = search_fts(args.query, args.k)

    if not hits:
        print("No matches.")
        return

    for i, h in enumerate(hits, start=1):
        preview = (h.text[:160] + "...") if len(h.text) > 160 else h.text
        print(
            f"{i}. score={h.score:.6f} "
            f"stable_id={h.stable_id} doc_id={h.document_id} chunk={h.chunk_index}"
        )
        print(f"   {preview}")
        print("")


if __name__ == "__main__":
    main()
