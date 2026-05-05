"""Prediction entrypoints for one or many submissions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .explanation_generator import format_review_text, generate_review
from .feature_engineering import build_feature_frame
from .risk_scoring import (
    calculate_risk_score,
    class_from_score,
    fraud_type_levels,
    recommendation,
    risk_level,
)
from .utils import MODELS_DIR


DEFAULT_MODEL_PATH = MODELS_DIR / "fraud_model.pkl"


def load_model_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any] | None:
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        return None


def _records_to_frame(submissions: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(submissions, pd.DataFrame):
        return submissions.copy()
    if isinstance(submissions, dict):
        return pd.DataFrame([submissions])
    return pd.DataFrame(list(submissions))


def _model_probabilities(bundle: dict[str, Any] | None, features: pd.DataFrame) -> list[dict[str, float] | None]:
    if not bundle:
        return [None for _ in range(len(features))]

    model = bundle.get("model")
    feature_columns = bundle.get("feature_columns", [])
    if model is None or not feature_columns:
        return [None for _ in range(len(features))]

    X = features.reindex(columns=feature_columns)
    try:
        probabilities = model.predict_proba(X)
        classes = [str(label).lower() for label in model.classes_]
        return [dict(zip(classes, row)) for row in probabilities]
    except Exception:
        return [None for _ in range(len(features))]


def predict_submissions(
    submissions: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    graph_df: pd.DataFrame | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> list[dict[str, Any]]:
    submission_frame = _records_to_frame(submissions)
    features = build_feature_frame(submission_frame, graph_df)
    bundle = load_model_bundle(model_path)
    probability_rows = _model_probabilities(bundle, features)

    results: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(features.iterrows()):
        probabilities = probability_rows[position] if position < len(probability_rows) else None
        score = calculate_risk_score(row, probabilities)
        predicted = class_from_score(score)
        if probabilities:
            model_class = max(probabilities, key=probabilities.get)
            if score < 25:
                predicted = "Clean"
            elif score >= 75:
                predicted = "Botted"
            elif model_class in {"botted", "suspicious"}:
                predicted = model_class.title()
            else:
                predicted = "Suspicious"

        level = risk_level(score)
        decision = recommendation(score, row)
        fraud_types = fraud_type_levels(row, score)
        review = generate_review(row, decision, score, level, predicted, fraud_types)

        result = {
            "decision": decision,
            "risk_score": score,
            "risk_level": level,
            "predicted_class": predicted,
            "fraud_types": fraud_types,
            "reasoning": review["reviewer_reasoning"],
            "creator_facing_reason": review["creator_facing_rejection_reason"],
            "review": review,
            "report": format_review_text(review),
        }
        if probabilities:
            result["model_probabilities"] = {key: round(float(value), 4) for key, value in probabilities.items()}
        results.append(result)
    return results


def predict_one(
    submission: dict[str, Any],
    graph_records: list[dict[str, Any]] | pd.DataFrame | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    graph_df = None
    if graph_records is not None:
        graph_df = graph_records if isinstance(graph_records, pd.DataFrame) else pd.DataFrame(graph_records)
    return predict_submissions(submission, graph_df=graph_df, model_path=model_path)[0]


def main() -> None:
    sample = {
        "platform": "TikTok",
        "views": 102146,
        "likes": 105,
        "comments": 0,
        "shares": 75,
        "graph_pattern": "flatline_then_vertical_spike",
        "traffic_source": None,
        "watch_time": None,
        "audience_location": None,
    }
    result = predict_one(sample)
    print(result["report"])


if __name__ == "__main__":
    main()
