from __future__ import annotations

from pathlib import Path

import joblib

from app.rerank.features import build_feature_vector

MODEL_PATH = Path("models/reranker.joblib")

_model = None


def _load_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"ML reranker model not found: {MODEL_PATH}. "
                "Train it using app.rerank.train."
            )

        _model = joblib.load(MODEL_PATH)

    return _model


def predict_ml_rerank_score(query: str, row: dict) -> float:
    model = _load_model()
    x = build_feature_vector(query, row)

    proba = model.predict_proba([x])[0]

    # Normal binary classification case
    if len(proba) >= 2:
        return float(proba[1])

    # Fallback: model was trained on only one class
    # If the only known class is 1, return 1.0
    # If the only known class is 0, return 0.0
    classes = getattr(model, "classes_", [])
    if len(classes) == 1:
        return 1.0 if int(classes[0]) == 1 else 0.0

    return float(proba[0])
