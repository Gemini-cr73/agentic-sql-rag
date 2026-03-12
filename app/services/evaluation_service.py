from __future__ import annotations

import json
from pathlib import Path
from typing import Any

APP_FILE = Path(__file__).resolve()
PROJECT_ROOT = APP_FILE.parent.parent.parent
EVAL_OUT_DIR = PROJECT_ROOT / "data" / "eval" / "out"
METRICS_FILE = EVAL_OUT_DIR / "metrics.json"
RANKING_ANALYSIS_FILE = EVAL_OUT_DIR / "ranking_analysis.json"


def _safe_read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _compute_average_grounding_from_runs(runs: list[dict[str, Any]]) -> float:
    """
    Approximate grounding from retrieved item scores when explicit grounding
    is not present in the evaluation files.
    Preference:
      1) average final_score
      2) average rerank_score
      3) average hybrid_score
      4) average base_score
    """
    collected: list[float] = []

    for run in runs:
        retrieved = run.get("retrieved")
        if not isinstance(retrieved, list):
            continue

        for item in retrieved:
            if not isinstance(item, dict):
                continue

            if "final_score" in item:
                collected.append(_safe_float(item.get("final_score")))
            elif "rerank_score" in item:
                collected.append(_safe_float(item.get("rerank_score")))
            elif "hybrid_score" in item:
                collected.append(_safe_float(item.get("hybrid_score")))
            elif "base_score" in item:
                collected.append(_safe_float(item.get("base_score")))

    if not collected:
        return 0.0

    return sum(collected) / len(collected)


def _extract_summary_from_metrics(metrics_data: Any) -> dict[str, Any]:
    """
    Supports:
    1) flat dict
    2) nested summary/metrics/aggregate/overall
    3) metrics.json with modes -> hybrid
    """
    empty = {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
        "average_grounding_score": 0.0,
        "rerank_comparison": {},
        "query_count": 0,
        "source": str(METRICS_FILE),
        "available": False,
    }

    if not isinstance(metrics_data, dict):
        return empty

    base = metrics_data

    for candidate_key in ("summary", "metrics", "aggregate", "overall"):
        candidate = metrics_data.get(candidate_key)
        if isinstance(candidate, dict):
            base = candidate
            break

    # Support the real file shape:
    # { "modes": { "hybrid": { ... } } }
    modes = metrics_data.get("modes")
    if isinstance(modes, dict):
        if isinstance(modes.get("hybrid"), dict):
            base = modes["hybrid"]
        else:
            for _, value in modes.items():
                if isinstance(value, dict):
                    base = value
                    break

    rerank_comparison = {}
    if metrics_data.get("rerank") is not None:
        rerank_comparison = {
            "enabled": bool(metrics_data.get("rerank", False)),
            "method": metrics_data.get("rerank_method", ""),
            "weight": _safe_float(metrics_data.get("rerank_weight", 0.0)),
            "top_k": _safe_int(metrics_data.get("rerank_top_k", 0)),
            "mode": "hybrid" if isinstance(modes, dict) and "hybrid" in modes else "",
        }

    return {
        "precision_at_k": _safe_float(
            base.get("precision_at_k", base.get("precision", 0.0))
        ),
        "recall_at_k": _safe_float(base.get("recall_at_k", base.get("recall", 0.0))),
        "mrr": _safe_float(base.get("mrr", 0.0)),
        "ndcg": _safe_float(base.get("ndcg", base.get("nDCG", 0.0))),
        "average_grounding_score": _safe_float(
            base.get("average_grounding_score", base.get("grounding_score", 0.0))
        ),
        "rerank_comparison": rerank_comparison,
        "query_count": _safe_int(base.get("query_count", base.get("num_queries", 0))),
        "source": str(METRICS_FILE),
        "available": True,
    }


def _extract_runs_from_ranking_analysis(ranking_data: Any) -> list[dict[str, Any]]:
    """
    Supports:
    1) list[dict]
    2) dict with runs / queries / results / items
    3) dict with runs=[{mode, per_query:[...]}]   <-- your actual file
    """
    raw_runs: list[Any] = []

    if isinstance(ranking_data, list):
        raw_runs = ranking_data
    elif isinstance(ranking_data, dict):
        if isinstance(ranking_data.get("runs"), list):
            raw_runs = ranking_data["runs"]
        elif isinstance(ranking_data.get("queries"), list):
            raw_runs = ranking_data["queries"]
        elif isinstance(ranking_data.get("results"), list):
            raw_runs = ranking_data["results"]
        elif isinstance(ranking_data.get("items"), list):
            raw_runs = ranking_data["items"]

    flattened_runs: list[dict[str, Any]] = []

    for idx, item in enumerate(raw_runs, start=1):
        if not isinstance(item, dict):
            continue

        # Support grouped structure: {"mode": "...", "per_query": [...]}
        if isinstance(item.get("per_query"), list):
            mode = str(item.get("mode", ""))
            for q_idx, per_query_item in enumerate(item["per_query"], start=1):
                if not isinstance(per_query_item, dict):
                    continue

                metrics = per_query_item.get("metrics", {})
                if not isinstance(metrics, dict):
                    metrics = {}

                flattened_runs.append(
                    {
                        "run_id": per_query_item.get("qid", f"{idx}-{q_idx}"),
                        "query": per_query_item.get("query", ""),
                        "precision_at_k": _safe_float(
                            metrics.get("precision_at_k", 0.0)
                        ),
                        "recall_at_k": _safe_float(metrics.get("recall_at_k", 0.0)),
                        "mrr": _safe_float(metrics.get("mrr", 0.0)),
                        "ndcg": _safe_float(metrics.get("ndcg", 0.0)),
                        "grounding_score": 0.0,
                        "rerank_enabled": bool(ranking_data.get("rerank", False)),
                        "latency_ms": _safe_float(
                            per_query_item.get("latency_ms", 0.0)
                        ),
                        "notes": f"mode={mode}" if mode else "",
                        "retrieved": per_query_item.get("retrieved", []),
                    }
                )
            continue

        # Flat fallback
        flattened_runs.append(
            {
                "run_id": item.get("run_id", item.get("id", idx)),
                "query": item.get(
                    "query", item.get("question", item.get("prompt", ""))
                ),
                "precision_at_k": _safe_float(
                    item.get("precision_at_k", item.get("precision", 0.0))
                ),
                "recall_at_k": _safe_float(
                    item.get("recall_at_k", item.get("recall", 0.0))
                ),
                "mrr": _safe_float(item.get("mrr", 0.0)),
                "ndcg": _safe_float(item.get("ndcg", item.get("nDCG", 0.0))),
                "grounding_score": _safe_float(
                    item.get(
                        "grounding_score", item.get("average_grounding_score", 0.0)
                    )
                ),
                "rerank_enabled": bool(
                    item.get("rerank_enabled", item.get("rerank", False))
                ),
                "latency_ms": _safe_float(item.get("latency_ms", 0.0)),
                "notes": str(item.get("notes", item.get("comment", ""))),
                "retrieved": item.get("retrieved", []),
            }
        )

    # If explicit grounding_score is missing, derive it from retrieved scores
    for run in flattened_runs:
        if _safe_float(run.get("grounding_score", 0.0)) == 0.0:
            derived = _compute_average_grounding_from_runs([run])
            run["grounding_score"] = derived

    return flattened_runs


def _compute_summary_from_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "average_grounding_score": 0.0,
            "rerank_comparison": {},
            "query_count": 0,
            "source": str(RANKING_ANALYSIS_FILE),
            "available": False,
        }

    query_count = len(runs)

    precision = sum(_safe_float(r.get("precision_at_k")) for r in runs) / query_count
    recall = sum(_safe_float(r.get("recall_at_k")) for r in runs) / query_count
    mrr = sum(_safe_float(r.get("mrr")) for r in runs) / query_count
    ndcg = sum(_safe_float(r.get("ndcg")) for r in runs) / query_count
    grounding = sum(_safe_float(r.get("grounding_score")) for r in runs) / query_count

    rerank_true = [r for r in runs if bool(r.get("rerank_enabled"))]
    rerank_false = [r for r in runs if not bool(r.get("rerank_enabled"))]

    rerank_comparison: dict[str, Any] = {}
    if rerank_true:
        rerank_comparison["enabled"] = {
            "count": len(rerank_true),
            "precision_at_k": sum(
                _safe_float(r.get("precision_at_k")) for r in rerank_true
            )
            / len(rerank_true),
            "recall_at_k": sum(_safe_float(r.get("recall_at_k")) for r in rerank_true)
            / len(rerank_true),
            "mrr": sum(_safe_float(r.get("mrr")) for r in rerank_true)
            / len(rerank_true),
            "ndcg": sum(_safe_float(r.get("ndcg")) for r in rerank_true)
            / len(rerank_true),
            "average_grounding_score": sum(
                _safe_float(r.get("grounding_score")) for r in rerank_true
            )
            / len(rerank_true),
        }

    if rerank_false:
        rerank_comparison["disabled"] = {
            "count": len(rerank_false),
            "precision_at_k": sum(
                _safe_float(r.get("precision_at_k")) for r in rerank_false
            )
            / len(rerank_false),
            "recall_at_k": sum(_safe_float(r.get("recall_at_k")) for r in rerank_false)
            / len(rerank_false),
            "mrr": sum(_safe_float(r.get("mrr")) for r in rerank_false)
            / len(rerank_false),
            "ndcg": sum(_safe_float(r.get("ndcg")) for r in rerank_false)
            / len(rerank_false),
            "average_grounding_score": sum(
                _safe_float(r.get("grounding_score")) for r in rerank_false
            )
            / len(rerank_false),
        }

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": mrr,
        "ndcg": ndcg,
        "average_grounding_score": grounding,
        "rerank_comparison": rerank_comparison,
        "query_count": query_count,
        "source": str(RANKING_ANALYSIS_FILE),
        "available": True,
    }


def _summary_is_effectively_empty(summary: dict[str, Any]) -> bool:
    numeric_fields = [
        _safe_float(summary.get("precision_at_k", 0.0)),
        _safe_float(summary.get("recall_at_k", 0.0)),
        _safe_float(summary.get("mrr", 0.0)),
        _safe_float(summary.get("ndcg", 0.0)),
        _safe_float(summary.get("average_grounding_score", 0.0)),
    ]
    query_count = _safe_int(summary.get("query_count", 0))
    return query_count == 0 or all(value == 0.0 for value in numeric_fields)


def get_evaluation_summary() -> dict[str, Any]:
    """
    Priority:
    1) Use metrics.json if it has real aggregate values
    2) Otherwise compute from ranking_analysis.json
    """
    metrics_data = _safe_read_json(METRICS_FILE)
    summary = _extract_summary_from_metrics(metrics_data)

    # If metrics.json has real values for the standard fields, use it.
    if not _summary_is_effectively_empty(summary):
        # If metrics lacks grounding, try to enrich it from ranking analysis.
        if _safe_float(summary.get("average_grounding_score", 0.0)) == 0.0:
            ranking_data = _safe_read_json(RANKING_ANALYSIS_FILE)
            runs = _extract_runs_from_ranking_analysis(ranking_data)
            if runs:
                summary["average_grounding_score"] = _compute_summary_from_runs(
                    runs
                ).get("average_grounding_score", 0.0)
        return summary

    ranking_data = _safe_read_json(RANKING_ANALYSIS_FILE)
    runs = _extract_runs_from_ranking_analysis(ranking_data)
    computed = _compute_summary_from_runs(runs)

    if computed.get("available"):
        return computed

    return summary


def get_evaluation_runs() -> dict[str, Any]:
    ranking_data = _safe_read_json(RANKING_ANALYSIS_FILE)
    runs = _extract_runs_from_ranking_analysis(ranking_data)

    return {
        "runs": [
            {
                "run_id": r["run_id"],
                "query": r["query"],
                "precision_at_k": r["precision_at_k"],
                "recall_at_k": r["recall_at_k"],
                "mrr": r["mrr"],
                "ndcg": r["ndcg"],
                "grounding_score": r["grounding_score"],
                "rerank_enabled": r["rerank_enabled"],
                "latency_ms": r["latency_ms"],
                "notes": r["notes"],
            }
            for r in runs
        ],
        "total": len(runs),
        "source": str(RANKING_ANALYSIS_FILE),
        "available": bool(runs),
    }
