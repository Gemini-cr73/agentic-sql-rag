# app/evaluation/run_eval.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.evaluation.metrics import compute_metrics_at_k
from app.rerank import rerank_rows
from app.retrieval.fts import fts_search
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector import vector_search


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_one_query(
    query: str,
    mode: str,
    k: int,
    alpha: float,
    *,
    rerank: bool = False,
    rerank_method: str = "overlap",
    rerank_weight: float = 0.15,
    rerank_top_k: int = 0,
) -> tuple[list[dict[str, Any]], list[str] | None, list[str] | None]:
    """
    Returns:
      hits_out: list of dicts with stable_id + scores for analysis
      pre_ids:  list of stable_ids before rerank (only for hybrid + rerank=True)
      post_ids: list of stable_ids after rerank (only for hybrid + rerank=True)

    We only rerank HYBRID mode here because the experiment spec is:
      query -> hybrid -> top-k -> rerank -> final ranking
    """
    pre_ids: list[str] | None = None
    post_ids: list[str] | None = None

    if mode == "fts":
        hits = fts_search(query, k=k)

    elif mode == "vector":
        hits = vector_search(query, k=k)

    elif mode == "hybrid":
        hits = hybrid_search(query, k=k, alpha=alpha)

        if rerank:
            pre_rows = [h.__dict__ if hasattr(h, "__dict__") else dict(h) for h in hits]

            reranked_rows, pre_ids, post_ids = rerank_rows(
                query,
                pre_rows,
                enabled=True,
                method=rerank_method,
                weight=float(rerank_weight),
                top_k=(int(rerank_top_k) or k),
            )

            hits = reranked_rows

    else:
        raise ValueError(f"Unknown mode: {mode}")

    out: list[dict[str, Any]] = []
    for h in hits:
        get = (
            (lambda key, default=None: h.get(key, default))
            if isinstance(h, dict)
            else None
        )

        stable_id = get("stable_id") if get else getattr(h, "stable_id", None)

        out.append(
            {
                "stable_id": stable_id,
                "hybrid_score": get("hybrid_score")
                if get
                else getattr(h, "hybrid_score", None),
                "fts_score": get("fts_score") if get else getattr(h, "fts_score", None),
                "vector_score": get("vector_score")
                if get
                else getattr(h, "vector_score", None),
                "rerank_method": get("rerank_method")
                if get
                else getattr(h, "rerank_method", None),
                "rerank_score": get("rerank_score")
                if get
                else getattr(h, "rerank_score", None),
                "base_score": get("base_score")
                if get
                else getattr(h, "base_score", None),
                "final_score": get("final_score")
                if get
                else getattr(h, "final_score", None),
            }
        )

    return out, pre_ids, post_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation over a small qrels set."
    )
    parser.add_argument(
        "--queries", default="data/eval/queries.json", help="Path to queries.json"
    )
    parser.add_argument(
        "--qrels", default="data/eval/qrels.json", help="Path to qrels.json"
    )
    parser.add_argument("--outdir", default="data/eval/out", help="Output directory")
    parser.add_argument("--k", type=int, default=5, help="K for @K metrics")
    parser.add_argument("--alpha", type=float, default=0.6, help="Hybrid alpha (0..1)")
    parser.add_argument(
        "--mode", choices=["fts", "vector", "hybrid", "all"], default="all"
    )

    parser.add_argument("--rerank", action="store_true", help="Enable reranking")
    parser.add_argument("--rerank-method", default="overlap")
    parser.add_argument("--rerank-weight", type=float, default=0.15)
    parser.add_argument("--rerank-top-k", type=int, default=0)

    args = parser.parse_args()

    queries_path = Path(args.queries)
    qrels_path = Path(args.qrels)
    outdir = Path(args.outdir)

    queries = _read_json(queries_path)
    qrels = _read_json(qrels_path)

    modes = ["fts", "vector", "hybrid"] if args.mode == "all" else [args.mode]

    metrics_summary: dict[str, Any] = {
        "k": args.k,
        "alpha": args.alpha,
        "modes": {},
        "rerank": bool(args.rerank),
        "rerank_method": str(args.rerank_method),
        "rerank_weight": float(args.rerank_weight),
        "rerank_top_k": int(args.rerank_top_k),
    }

    ranking_analysis: dict[str, Any] = {
        "k": args.k,
        "alpha": args.alpha,
        "rerank": bool(args.rerank),
        "rerank_method": str(args.rerank_method),
        "rerank_weight": float(args.rerank_weight),
        "rerank_top_k": int(args.rerank_top_k),
        "runs": [],
    }

    for mode in modes:
        per_query = []
        agg = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
        n = 0

        for q in queries:
            qid = q["qid"]
            query_text = q["query"]
            graded = qrels.get(qid, {})
            relevant_binary = {doc_id for doc_id, rel in graded.items() if int(rel) > 0}
            relevant_graded = {doc_id: int(rel) for doc_id, rel in graded.items()}

            hits, pre_ids, post_ids = _run_one_query(
                query_text,
                mode=mode,
                k=args.k,
                alpha=args.alpha,
                rerank=bool(args.rerank),
                rerank_method=str(args.rerank_method),
                rerank_weight=float(args.rerank_weight),
                rerank_top_k=int(args.rerank_top_k),
            )
            retrieved_ids = [h["stable_id"] for h in hits if h.get("stable_id")]

            m = compute_metrics_at_k(
                retrieved_ids=retrieved_ids,
                relevant_binary=relevant_binary,
                relevant_graded=relevant_graded,
                k=args.k,
            )

            item: dict[str, Any] = {
                "qid": qid,
                "query": query_text,
                "metrics": asdict(m),
                "retrieved": hits,
                "relevant": relevant_graded,
            }

            if (
                mode == "hybrid"
                and args.rerank
                and pre_ids is not None
                and post_ids is not None
            ):
                item["pre_rerank_ids"] = pre_ids
                item["post_rerank_ids"] = post_ids

            per_query.append(item)

            agg["precision"] += m.precision_at_k
            agg["recall"] += m.recall_at_k
            agg["mrr"] += m.mrr
            agg["ndcg"] += m.ndcg
            n += 1

        if n:
            metrics_summary["modes"][mode] = {
                "precision_at_k": agg["precision"] / n,
                "recall_at_k": agg["recall"] / n,
                "mrr": agg["mrr"] / n,
                "ndcg": agg["ndcg"] / n,
                "num_queries": n,
            }
        else:
            metrics_summary["modes"][mode] = {"num_queries": 0}

        ranking_analysis["runs"].append({"mode": mode, "per_query": per_query})

    _write_json(outdir / "metrics.json", metrics_summary)
    _write_json(outdir / "ranking_analysis.json", ranking_analysis)

    print(f"[OK] Wrote: {outdir / 'metrics.json'}")
    print(f"[OK] Wrote: {outdir / 'ranking_analysis.json'}")


if __name__ == "__main__":
    main()
