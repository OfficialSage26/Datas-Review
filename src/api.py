"""FastAPI app for AI video fraud review."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from .predict import predict_one, predict_submissions
from .train_model import train_model


app = FastAPI(title="AI Video Fraud Review API", version="1.0.0")


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "AI Video Fraud Review API",
        "endpoints": ["/predict", "/predict-batch", "/retrain"],
    }


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    graph_records = payload.pop("graph_records", None)
    return predict_one(payload, graph_records=graph_records)


@app.post("/predict-batch")
def predict_batch(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        items = payload.get("items") or payload.get("submissions") or []
    else:
        items = payload
    return {"count": len(items), "results": predict_submissions(items)}


@app.post("/retrain")
def retrain() -> dict[str, Any]:
    metrics = train_model(generate_if_missing=True)
    return {"status": "trained", "metrics": metrics}

