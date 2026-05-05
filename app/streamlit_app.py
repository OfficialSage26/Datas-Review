"""Streamlit dashboard for AI video fraud review."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_one, predict_submissions
from src.train_model import METRICS_PATH


st.set_page_config(page_title="AI Video Fraud Review", layout="wide")
st.title("AI Video Fraud Review")

tab_manual, tab_batch, tab_model = st.tabs(["Manual Review", "CSV Batch", "Model"])

with tab_manual:
    col_left, col_right = st.columns(2)
    with col_left:
        platform = st.selectbox("Platform", ["TikTok", "Instagram", "YouTube Shorts"])
        views = st.number_input("Views", min_value=0, value=102146, step=100)
        likes = st.number_input("Likes", min_value=0, value=105, step=10)
        comments = st.number_input("Comments", min_value=0, value=0, step=1)
        shares = st.number_input("Shares", min_value=0, value=75, step=1)
    with col_right:
        graph_pattern = st.selectbox(
            "Graph pattern",
            [
                "unknown",
                "smooth_gradual",
                "normal_viral_smooth",
                "low_engagement_smooth",
                "late_like_spike",
                "likes_freeze",
                "step_like_batches",
                "flat_then_spike",
                "vertical_spike_no_engagement",
            ],
            index=7,
        )
        avg_views_30d = st.number_input("Average views last 30 days", min_value=0, value=2176, step=100)
        avg_watch_time_sec = st.number_input("Average watch time seconds", min_value=0.0, value=0.0, step=0.1)
        completion_rate = st.number_input("Completion rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        comment_quality = st.selectbox(
            "Comment quality",
            ["unknown", "relevant_discussion", "short_but_relevant", "mixed", "generic_repeated", "emoji_spam", "unrelated", "none"],
        )

    if st.button("Review Submission", type="primary"):
        result = predict_one(
            {
                "platform": platform,
                "views": views,
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "graph_pattern": graph_pattern,
                "avg_views_30d": avg_views_30d,
                "avg_watch_time_sec": avg_watch_time_sec,
                "completion_rate": completion_rate,
                "comment_quality": comment_quality,
            }
        )
        st.metric("Decision", result["decision"])
        st.metric("Fraud Risk Score", result["risk_score"])
        st.metric("Risk Level", result["risk_level"])
        st.write("Predicted class:", result["predicted_class"])
        st.write("Graph analysis:", result["review"]["graph_analysis"])
        st.write("Suspicious signals:")
        st.write(result["review"]["suspicious_signals"])
        st.write("Creator-facing reason:")
        st.info(result["creator_facing_reason"])
        st.text(result["report"])

with tab_batch:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        st.dataframe(frame.head(20))
        if st.button("Review CSV"):
            predictions = predict_submissions(frame)
            rows = []
            for index, result in enumerate(predictions):
                rows.append(
                    {
                        "row": index,
                        "decision": result["decision"],
                        "risk_score": result["risk_score"],
                        "risk_level": result["risk_level"],
                        "predicted_class": result["predicted_class"],
                        "creator_facing_reason": result["creator_facing_reason"],
                    }
                )
            st.dataframe(pd.DataFrame(rows))
            st.json(predictions[:3])

with tab_model:
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.write("Rows trained:", metrics.get("rows"))
        st.write("Accuracy:", metrics.get("accuracy"))
        st.write("Synthetic note:", metrics.get("synthetic_training_note"))
        st.write("Confusion matrix labels:", metrics.get("labels"))
        st.dataframe(pd.DataFrame(metrics.get("confusion_matrix", []), columns=metrics.get("labels", [])))
        importance = pd.DataFrame(metrics.get("feature_importance", []))
        if not importance.empty:
            st.subheader("Feature Importance")
            st.dataframe(importance)
    else:
        st.warning("No trained model metrics found yet. Run: python -m src.train_model")

