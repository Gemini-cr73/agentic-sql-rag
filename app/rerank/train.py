from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.rerank.features import build_feature_vector
from app.retrieval.hybrid import hybrid_search


def load_json(path: str):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--k", type=int, default=20)

    args = parser.parse_args()

    queries = load_json(args.queries)
    qrels = load_json(args.qrels)

    X = []
    y = []

    for q in queries:
        query = q["query"]
        qid = q["qid"]

        rel_docs = qrels.get(qid, {})

        hits = hybrid_search(query, k=args.k)

        for h in hits:
            row = h.__dict__

            features = build_feature_vector(query, row)

            label = 1 if row["stable_id"] in rel_docs else 0

            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("Training samples:", len(X))
    print("Positive labels:", int((y == 1).sum()))
    print("Negative labels:", int((y == 0).sum()))

    unique_labels = set(int(v) for v in y.tolist())
    if len(unique_labels) < 2:
        raise RuntimeError(
            "ML reranker training needs both positive and negative examples. "
            "Your qrels/retrieval results currently produced only one class. "
            "Add more queries or ensure retrieved results include both relevant and non-relevant items."
        )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
    )

    model.fit(X, y)

    Path("models").mkdir(exist_ok=True)

    joblib.dump(model, "models/reranker.joblib")

    print("Model saved → models/reranker.joblib")


if __name__ == "__main__":
    main()
