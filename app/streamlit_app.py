"""Streamlit dashboard for AI video fraud review."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_one, predict_submissions
from src.moderation import REJECT_FULL_ANALYTICS
from src.screenshot_analyzer import analyze_dashboard_screenshot
from src.train_model import METRICS_PATH


st.set_page_config(page_title="AI Video Fraud Review", layout="wide")
st.title("AI Video Fraud Review")

tab_screenshot, tab_manual, tab_batch, tab_model = st.tabs(["Screenshot Review", "Manual Review", "CSV Batch", "Model"])

MAX_SCREENSHOT_UPLOAD_BYTES = 15 * 1024 * 1024


def show_moderation_result(result: dict) -> None:
    moderation = result.get("moderation") or {}
    decision = moderation.get("decision", result.get("decision", "Reject: Send Full Analytics"))
    confidence = moderation.get("confidence", "Low")
    if decision == "Approved":
        st.success(f"Decision: {decision}")
    else:
        st.error(f"Decision: {decision}")
    st.subheader(f"Confidence: {confidence}")
    st.markdown("**Reason:**")
    for item in moderation.get("reason", []):
        st.markdown(f"- {item}")
    st.markdown("**Red Flags:**")
    for item in moderation.get("red_flags", ["No major red flags found."]):
        st.markdown(f"- {item}")
    st.markdown("**Mod Note:**")
    st.info(moderation.get("mod_note", result.get("creator_facing_reason", "")))


def safe_json(value: object) -> object:
    """Convert model output containing numpy/pandas values into Streamlit-safe JSON."""
    return json.loads(json.dumps(value, default=str))


def show_fail_closed_upload_error(error: Exception) -> None:
    """Keep the reviewer workflow visible when screenshot analysis fails."""
    show_moderation_result(
        {
            "moderation": {
                "decision": REJECT_FULL_ANALYTICS,
                "confidence": "Low",
                "reason": ["Screenshot analysis did not complete, so the submission cannot be approved from visible proof."],
                "red_flags": ["Uploaded screenshot could not be fully analyzed."],
                "mod_note": "Please send full analytics before this submission can be approved.",
            }
        }
    )
    with st.expander("Technical upload error"):
        st.exception(error)


with tab_screenshot:
    st.subheader("Upload Submission Screenshot")
    platform_from_reviewer = st.selectbox(
        "Platform",
        ["TikTok", "Instagram", "YouTube Shorts", "Unknown"],
        key="screenshot_platform",
    )
    campaign_requirements = st.text_area(
        "Campaign requirements shown or expected",
        placeholder="Optional. If requirements cannot be confirmed from the screenshot, the review will request full analytics.",
        key="screenshot_campaign_requirements",
    )
    uploaded_image = st.file_uploader(
        "Upload screenshot",
        type=["png", "jpg", "jpeg"],
        key="screenshot_upload",
    )
    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()
        st.image(image_bytes, caption="Uploaded screenshot", use_container_width=True)

        if len(image_bytes) > MAX_SCREENSHOT_UPLOAD_BYTES:
            st.warning("This screenshot is large and may take longer to analyze. Crop to the visible submission card if the review is slow.")

        analyze_now = st.button("Analyze Uploaded Screenshot", type="primary", key="analyze_screenshot_button")
        if not analyze_now:
            st.info("Preview loaded. Click Analyze Uploaded Screenshot to run OCR, graph vision, and fraud review.")
        else:
            suffix = Path(uploaded_image.name).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg"}:
                suffix = ".png"
            temp_path = None
            try:
                with st.spinner("Analyzing screenshot with OCR, graph vision, and the fraud model..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(image_bytes)
                        temp_path = Path(temp_file.name)
                    analysis_result = analyze_dashboard_screenshot(
                        temp_path,
                        platform=None if platform_from_reviewer == "Unknown" else platform_from_reviewer,
                        extra_fields={"campaign_requirements": campaign_requirements.strip()} if campaign_requirements.strip() else None,
                    )
            except Exception as exc:
                show_fail_closed_upload_error(exc)
                analysis_result = None
            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

            if analysis_result:
                display_result = dict(analysis_result.get("prediction") or {})
                display_result["moderation"] = analysis_result.get("moderation") or {}
                show_moderation_result(display_result)

                analysis = analysis_result["screenshot_analysis"]
                fields = analysis.get("fields", {})
                graph_vision = analysis.get("graph_vision", {})
                st.markdown("**Extracted Screenshot Fields:**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"field": "platform", "value": fields.get("platform") or analysis_result["submission"].get("platform")},
                            {"field": "views", "value": fields.get("views")},
                            {"field": "likes", "value": fields.get("likes")},
                            {"field": "comments", "value": fields.get("comments")},
                            {"field": "shares", "value": fields.get("shares")},
                            {"field": "status", "value": fields.get("status")},
                            {"field": "graph_pattern", "value": analysis.get("graph_shape")},
                            {"field": "ocr_available", "value": analysis.get("ocr_available")},
                            {"field": "cv_available", "value": analysis.get("cv_available")},
                            {"field": "graph_confidence", "value": graph_vision.get("confidence")},
                            {"field": "visible_action", "value": graph_vision.get("action_visible")},
                        ]
                    ),
                    use_container_width=True,
                )
                with st.expander("Internal model details"):
                    prediction = analysis_result["prediction"]
                    st.write("Risk score:", prediction["risk_score"])
                    st.write("Risk level:", prediction["risk_level"])
                    st.write("Predicted class:", prediction["predicted_class"])
                    st.write("Graph analysis:", prediction["review"]["graph_analysis"])
                    st.write("Suspicious signals:", prediction["review"]["suspicious_signals"])
                    st.write("Missing evidence:", prediction["review"]["missing_evidence_needed"])
                    if st.checkbox("Show raw screenshot analysis JSON"):
                        st.json(safe_json(analysis_result))

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
        show_moderation_result(result)
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
                        "decision": result["moderation"]["decision"],
                        "confidence": result["moderation"]["confidence"],
                        "risk_score": result["risk_score"],
                        "risk_level": result["risk_level"],
                        "predicted_class": result["predicted_class"],
                        "mod_note": result["moderation"]["mod_note"],
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
