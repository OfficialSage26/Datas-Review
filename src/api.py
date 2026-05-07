"""FastAPI app for AI video fraud review."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from .predict import predict_one, predict_submissions
from .tikfly_client import TikflyError, fetch_tiktok_review_context, get_env_rapidapi_key
from .train_model import train_model


app = FastAPI(title="AI Video Fraud Review API", version="1.0.0")


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "AI Video Fraud Review API",
        "endpoints": ["/predict", "/predict-batch", "/predict-tiktok-url", "/retrain"],
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


@app.post("/predict-tiktok-url")
def predict_tiktok_url(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = get_env_rapidapi_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="RAPIDAPI_KEY is not configured on the server.")

    url = str(payload.get("url") or payload.get("video_url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Request body must include a TikTok video URL.")

    try:
        context = fetch_tiktok_review_context(
            url,
            api_key=api_key,
            history_count=payload.get("history_count", 35),
        )
    except TikflyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prediction = predict_one(context["submission"])
    return {
        **context,
        "prediction": prediction,
        "moderation": prediction["moderation"],
    }


@app.post("/retrain")
def retrain() -> dict[str, Any]:
    metrics = train_model(generate_if_missing=True)
    return {"status": "trained", "metrics": metrics}
