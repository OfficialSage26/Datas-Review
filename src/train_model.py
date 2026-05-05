"""Train rule-based and ML fraud classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import GENERATED_SUBMISSION_CSV, load_all
from .feature_engineering import build_feature_frame
from .risk_scoring import calculate_rule_score, class_from_score
from .synthetic_data_generator import generate_synthetic_data
from .utils import DATAS_DIR, MODELS_DIR


MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
METRICS_PATH = MODELS_DIR / "training_metrics.json"

LEAKAGE_COLUMNS = {
    "submission_id",
    "label",
    "fraud_risk_score",
    "risk_level",
    "decision_recommendation",
    "likely_fraud_type",
    "reviewer_reason",
    "suggested_action",
    "source_type",
    "source_note",
}


def ensure_synthetic_data(min_rows: int = 10000) -> None:
    generated_path = DATAS_DIR / GENERATED_SUBMISSION_CSV
    if generated_path.exists():
        try:
            if len(pd.read_csv(generated_path, usecols=["submission_id"])) >= min_rows:
                return
        except Exception:
            pass
    generate_synthetic_data(n_rows=min_rows)


def _make_encoder():
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _feature_importance(model: Any, numeric_columns: list[str], categorical_columns: list[str]) -> list[dict[str, Any]]:
    try:
        preprocessor = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]
        cat_names: list[str] = []
        if categorical_columns:
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_names = list(cat_encoder.get_feature_names_out(categorical_columns))
        names = list(numeric_columns) + cat_names
        importances = estimator.feature_importances_
        pairs = sorted(zip(names, importances), key=lambda item: item[1], reverse=True)
        return [{"feature": name, "importance": round(float(value), 6)} for name, value in pairs[:40]]
    except Exception:
        return []


def train_model(generate_if_missing: bool = True) -> dict[str, Any]:
    if generate_if_missing:
        ensure_synthetic_data()

    submissions, graph, rules = load_all(include_generated=True)
    features = build_feature_frame(submissions, graph)
    if "label" not in features.columns:
        raise ValueError("Training data must include a label column.")

    y = features["label"].astype(str).str.lower()
    X = features.drop(columns=[column for column in LEAKAGE_COLUMNS if column in features.columns])

    numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.22,
        random_state=42,
        stratify=y,
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _make_encoder()),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=260,
        random_state=42,
        class_weight={"clean": 1.0, "suspicious": 2.0, "botted": 2.5},
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model = Pipeline(steps=[("preprocess", preprocess), ("model", classifier)])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rule_pred = [class_from_score(calculate_rule_score(row)).lower() for _, row in X_test.iterrows()]

    labels = ["clean", "suspicious", "botted"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    rule_precision, rule_recall, rule_f1, _ = precision_recall_fscore_support(
        y_test,
        rule_pred,
        labels=labels,
        zero_division=0,
    )

    metrics = {
        "rows": int(len(features)),
        "original_plus_generated": True,
        "synthetic_training_note": "Generated rows supplement training only and are not production validation evidence.",
        "guide_rule_count": len(rules),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "labels": labels,
        "per_class": {
            label: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "rule_based_baseline": {
            "accuracy": round(float(accuracy_score(y_test, rule_pred)), 4),
            "per_class": {
                label: {
                    "precision": round(float(rule_precision[index]), 4),
                    "recall": round(float(rule_recall[index]), 4),
                    "f1": round(float(rule_f1[index]), 4),
                }
                for index, label in enumerate(labels)
            },
            "confusion_matrix": confusion_matrix(y_test, rule_pred, labels=labels).tolist(),
        },
        "feature_importance": _feature_importance(model, numeric_columns, categorical_columns),
    }

    bundle = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "classes": labels,
        "metrics": metrics,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Rule-based baseline:")
    print(json.dumps(metrics["rule_based_baseline"], indent=2))
    print("\nMachine learning classifier:")
    print(f"Accuracy: {metrics['accuracy']}")
    print(metrics["classification_report"])
    print("Confusion matrix labels:", labels)
    print(np.array(metrics["confusion_matrix"]))
    print("\nTop feature importance:")
    for item in metrics["feature_importance"][:15]:
        print(f"{item['feature']}: {item['importance']}")
    print(f"\nSaved model: {MODEL_PATH}")
    return metrics


def main() -> None:
    train_model(generate_if_missing=True)


if __name__ == "__main__":
    main()

