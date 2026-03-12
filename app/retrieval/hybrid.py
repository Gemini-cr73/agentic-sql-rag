# app/retrieval/hybrid.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import text

from app.db.session import new_session
from app.embeddings.stub import embed_text

log = logging.getLogger(__name__)

MergeStrategy = Literal["union", "intersection", "rrf"]


@dataclass(frozen=True)
class RetrievalHit:
    stable_id: str
    chunk_id: int
    document_id: int
    title: str | None
    source: str | None
    chunk_index: int
    char_start: int
    char_end: int
    content: str

    # raw channel scores
    fts_score: float
    vector_distance: float | None
    vector_score: float

    # normalized + combined score
    hybrid_score: float


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_max(values: list[float]) -> float:
    m = 0.0
    for v in values:
        if v > m:
            m = v
    return m


def _normalize_max(values_by_id: dict[str, float]) -> dict[str, float]:
    """
    Stable normalization: v_norm = v / max(v)
    If max == 0, all normalize to 0.
    """
    mx = _safe_max(list(values_by_id.values()))
    if mx <= 0.0:
        return {k: 0.0 for k in values_by_id}
    return {k: (float(v) / mx) for k, v in values_by_id.items()}


def fts_search(query: str, k: int) -> list[dict[str, Any]]:
    """
    FTS search over doc_chunks.tsv (tsvector).
    Deterministic ordering: fts_score DESC, stable_id ASC
    """
    sql = text(
        """
        WITH q AS (
            SELECT plainto_tsquery('english', :q) AS tsq
        )
        SELECT
            c.stable_id AS stable_id,
            c.id AS chunk_id,
            c.document_id AS document_id,
            d.title AS title,
            d.source AS source,
            c.chunk_index AS chunk_index,
            c.char_start AS char_start,
            c.char_end AS char_end,
            c.text AS content,
            ts_rank(COALESCE(c.tsv, ''::tsvector), q.tsq) AS fts_score
        FROM doc_chunks c
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN q
        WHERE COALESCE(c.tsv, ''::tsvector) @@ q.tsq
        ORDER BY fts_score DESC, c.stable_id ASC
        LIMIT :k
        """
    )

    with new_session() as s:
        rows = s.execute(sql, {"q": query, "k": int(k)}).mappings().all()
        return [dict(r) for r in rows]


def vector_search(query: str, k: int) -> list[dict[str, Any]]:
    """
    Vector similarity search over doc_chunks.embedding (pgvector).
    Deterministic ordering: distance ASC, stable_id ASC
    """
    vec = embed_text(query, dim=384)
    vec_lit = _vec_literal(vec)

    sql = text(
        """
        SELECT
            c.stable_id AS stable_id,
            c.id AS chunk_id,
            c.document_id AS document_id,
            d.title AS title,
            d.source AS source,
            c.chunk_index AS chunk_index,
            c.char_start AS char_start,
            c.char_end AS char_end,
            c.text AS content,
            (c.embedding <-> CAST(:vec AS vector)) AS distance
        FROM doc_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
        ORDER BY distance ASC, c.stable_id ASC
        LIMIT :k
        """
    )

    with new_session() as s:
        rows = s.execute(sql, {"vec": vec_lit, "k": int(k)}).mappings().all()

    out: list[dict[str, Any]] = []
    for r in rows:
        dist = float(r["distance"])
        vec_score = 1.0 / (1.0 + dist)

        out.append(
            {
                "stable_id": str(r["stable_id"]),
                "chunk_id": int(r["chunk_id"]),
                "document_id": int(r["document_id"]),
                "title": r.get("title"),
                "source": r.get("source"),
                "chunk_index": int(r["chunk_index"]),
                "char_start": int(r["char_start"]),
                "char_end": int(r["char_end"]),
                "content": str(r["content"]),
                "vector_distance": dist,
                "vector_score": vec_score,
            }
        )
    return out


def hybrid_search(
    query: str,
    k: int = 5,
    alpha: float = 0.6,
    use_fts: bool = True,
    use_vector: bool = True,
    merge_strategy: MergeStrategy = "union",
) -> list[RetrievalHit]:
    """
    Hybrid retrieval combining:
      - FTS score
      - Vector similarity score

    Additions:
      - score normalization
      - configurable merge strategy: union | intersection | rrf
      - deterministic sorting
    """
    if not query or not str(query).strip():
        return []

    k = int(k) if int(k) > 0 else 5
    if k > 100:
        k = 100

    alpha = _clamp01(float(alpha))

    if not use_fts and not use_vector:
        return []

    k_fetch = max(k, 10)
    fts_rows = fts_search(query, k_fetch) if use_fts else []
    vec_rows = vector_search(query, k_fetch) if use_vector else []

    if not fts_rows and not vec_rows:
        return []

    fts_rank: dict[str, int] = {}
    vec_rank: dict[str, int] = {}

    for i, r in enumerate(fts_rows, start=1):
        sid = str(r["stable_id"])
        if sid not in fts_rank:
            fts_rank[sid] = i

    for i, r in enumerate(vec_rows, start=1):
        sid = str(r["stable_id"])
        if sid not in vec_rank:
            vec_rank[sid] = i

    merged: dict[str, dict[str, Any]] = {}

    for r in fts_rows:
        sid = str(r["stable_id"])
        merged[sid] = {
            "stable_id": sid,
            "chunk_id": int(r["chunk_id"]),
            "document_id": int(r["document_id"]),
            "title": r.get("title"),
            "source": r.get("source"),
            "chunk_index": int(r["chunk_index"]),
            "char_start": int(r["char_start"]),
            "char_end": int(r["char_end"]),
            "content": str(r["content"]),
            "fts_score": float(r.get("fts_score") or 0.0),
            "vector_distance": None,
            "vector_score": 0.0,
        }

    for r in vec_rows:
        sid = str(r["stable_id"])
        if sid not in merged:
            merged[sid] = {
                "stable_id": sid,
                "chunk_id": int(r["chunk_id"]),
                "document_id": int(r["document_id"]),
                "title": r.get("title"),
                "source": r.get("source"),
                "chunk_index": int(r["chunk_index"]),
                "char_start": int(r["char_start"]),
                "char_end": int(r["char_end"]),
                "content": str(r["content"]),
                "fts_score": 0.0,
                "vector_distance": (
                    float(r.get("vector_distance"))
                    if r.get("vector_distance") is not None
                    else None
                ),
                "vector_score": float(r.get("vector_score") or 0.0),
            }
        else:
            merged[sid]["vector_distance"] = (
                float(r.get("vector_distance"))
                if r.get("vector_distance") is not None
                else None
            )
            merged[sid]["vector_score"] = float(r.get("vector_score") or 0.0)

        # fill any missing metadata from vector side
        if not merged[sid].get("title") and r.get("title"):
            merged[sid]["title"] = r.get("title")
        if not merged[sid].get("source") and r.get("source"):
            merged[sid]["source"] = r.get("source")

    if merge_strategy == "intersection":
        keep = set(fts_rank.keys()) & set(vec_rank.keys())
        merged = {sid: row for sid, row in merged.items() if sid in keep}
        if not merged:
            return []

    fts_by_id = {sid: float(v["fts_score"]) for sid, v in merged.items()}
    vec_by_id = {sid: float(v["vector_score"]) for sid, v in merged.items()}
    norm_fts = _normalize_max(fts_by_id)
    norm_vec = _normalize_max(vec_by_id)

    hits: list[RetrievalHit] = []

    if merge_strategy == "rrf":
        k0 = 60.0
        w_fts = alpha
        w_vec = 1.0 - alpha

        for sid, v in merged.items():
            r_fts = float(fts_rank.get(sid, 10_000))
            r_vec = float(vec_rank.get(sid, 10_000))
            rrf_score = (w_fts / (k0 + r_fts)) + (w_vec / (k0 + r_vec))

            hits.append(
                RetrievalHit(
                    stable_id=sid,
                    chunk_id=int(v["chunk_id"]),
                    document_id=int(v["document_id"]),
                    title=v.get("title"),
                    source=v.get("source"),
                    chunk_index=int(v["chunk_index"]),
                    char_start=int(v["char_start"]),
                    char_end=int(v["char_end"]),
                    content=str(v["content"]),
                    fts_score=float(v["fts_score"]),
                    vector_distance=v.get("vector_distance"),
                    vector_score=float(v["vector_score"]),
                    hybrid_score=float(rrf_score),
                )
            )
    else:
        for sid, v in merged.items():
            hybrid = alpha * float(norm_fts.get(sid, 0.0)) + (1.0 - alpha) * float(
                norm_vec.get(sid, 0.0)
            )

            hits.append(
                RetrievalHit(
                    stable_id=sid,
                    chunk_id=int(v["chunk_id"]),
                    document_id=int(v["document_id"]),
                    title=v.get("title"),
                    source=v.get("source"),
                    chunk_index=int(v["chunk_index"]),
                    char_start=int(v["char_start"]),
                    char_end=int(v["char_end"]),
                    content=str(v["content"]),
                    fts_score=float(v["fts_score"]),
                    vector_distance=v.get("vector_distance"),
                    vector_score=float(v["vector_score"]),
                    hybrid_score=float(hybrid),
                )
            )

    hits.sort(key=lambda h: (-h.hybrid_score, h.stable_id))
    log.debug(
        "hybrid_search query=%r k=%s alpha=%.3f use_fts=%s use_vector=%s strategy=%s returned=%s",
        query,
        k,
        alpha,
        use_fts,
        use_vector,
        merge_strategy,
        len(hits[:k]),
    )
    return hits[:k]
