from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.feature_engineering import build_feature_frame, create_graph_features
from src.moderation import moderation_decision_from_prediction
from src.predict import predict_one
from src.risk_scoring import calculate_risk_score, risk_level
from src.screenshot_analyzer import apply_metric_overrides, extract_visible_metrics, extracted_metrics_to_submission


ROOT = Path(__file__).resolve().parents[1]


def _png_bytes(color: str = "#222222") -> bytes:
    image = Image.new("RGB", (320, 180), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_ratio_and_graph_features_with_missing_optional_fields() -> None:
    submissions = pd.DataFrame(
        [
            {
                "submission_id": "T-1",
                "views": 10000,
                "likes": 50,
                "comments": 0,
                "shares": 1,
                "graph_pattern": "flat_then_spike",
            }
        ]
    )
    graph = pd.DataFrame(
        [
            {"submission_id": "T-1", "hour_since_post": 0, "views_cumulative": 0, "likes_cumulative": 0, "comments_cumulative": 0, "shares_cumulative": 0},
            {"submission_id": "T-1", "hour_since_post": 2, "views_cumulative": 0, "likes_cumulative": 0, "comments_cumulative": 0, "shares_cumulative": 0},
            {"submission_id": "T-1", "hour_since_post": 4, "views_cumulative": 9800, "likes_cumulative": 1, "comments_cumulative": 0, "shares_cumulative": 0},
            {"submission_id": "T-1", "hour_since_post": 6, "views_cumulative": 10000, "likes_cumulative": 50, "comments_cumulative": 0, "shares_cumulative": 1},
        ]
    )
    features = build_feature_frame(submissions, graph)
    row = features.iloc[0]
    assert row["like_rate"] == 0.005
    assert row["comment_rate"] == 0
    assert row["high_views_zero_comments"] == 1
    assert row["max_view_jump"] >= 9800
    assert row["graph_suspicion_score"] >= 80


def test_create_graph_features_handles_empty_graph() -> None:
    assert create_graph_features(pd.DataFrame()).empty


def test_risk_level_boundaries() -> None:
    assert risk_level(0) == "Low"
    assert risk_level(24) == "Low"
    assert risk_level(25) == "Medium"
    assert risk_level(50) == "High"
    assert risk_level(75) == "Critical"


def test_high_view_zero_comment_prediction_is_high_risk() -> None:
    result = predict_one(
        {
            "platform": "TikTok",
            "views": 102146,
            "likes": 105,
            "comments": 0,
            "shares": 75,
            "graph_pattern": "vertical_spike_no_engagement",
            "avg_views_30d": 2176,
            "completion_rate": 0.0574,
        }
    )
    assert result["risk_score"] >= 75
    assert result["risk_level"] == "Critical"
    assert result["predicted_class"] == "Botted"
    assert result["decision"] in {"Reject", "Escalate"}
    assert "fake_views" in result["fraud_types"]
    assert result["creator_facing_reason"]
    assert "AI Video Fraud Review" in result["report"]


def test_rule_score_clean_sample_is_low() -> None:
    frame = build_feature_frame(
        pd.DataFrame(
            [
                {
                    "views": 50000,
                    "likes": 3500,
                    "comments": 220,
                    "shares": 300,
                    "graph_pattern": "smooth_gradual",
                    "completion_rate": 0.55,
                    "avg_views_30d": 12000,
                }
            ]
        )
    )
    assert calculate_risk_score(frame.iloc[0]) < 25


def test_stronger_graph_features_detect_flatline_spike_and_late_like() -> None:
    graph = pd.DataFrame(
        [
            {"submission_id": "G-1", "hour_since_post": 0, "views_cumulative": 0, "likes_cumulative": 0, "comments_cumulative": 0, "shares_cumulative": 0},
            {"submission_id": "G-1", "hour_since_post": 2, "views_cumulative": 10, "likes_cumulative": 0, "comments_cumulative": 0, "shares_cumulative": 0},
            {"submission_id": "G-1", "hour_since_post": 4, "views_cumulative": 11, "likes_cumulative": 0, "comments_cumulative": 0, "shares_cumulative": 0},
            {"submission_id": "G-1", "hour_since_post": 6, "views_cumulative": 12, "likes_cumulative": 0, "comments_cumulative": 0, "shares_cumulative": 0},
            {"submission_id": "G-1", "hour_since_post": 8, "views_cumulative": 9000, "likes_cumulative": 5, "comments_cumulative": 0, "shares_cumulative": 0},
            {"submission_id": "G-1", "hour_since_post": 10, "views_cumulative": 10000, "likes_cumulative": 5, "comments_cumulative": 0, "shares_cumulative": 0},
            {"submission_id": "G-1", "hour_since_post": 12, "views_cumulative": 10200, "likes_cumulative": 450, "comments_cumulative": 0, "shares_cumulative": 0},
        ]
    )
    features = create_graph_features(graph).iloc[0]
    assert features["vertical_spike_flag"] == 1
    assert features["pre_spike_flatline_hours"] >= 4
    assert features["flatline_then_spike_score"] >= 55
    assert features["views_engagement_divergence"] > 0
    assert features["late_like_spike"] == 1
    assert features["view_like_peak_lag_hours"] >= 4


def test_ocr_text_metric_extraction_supports_label_before_and_after() -> None:
    fields = extract_visible_metrics("Views: 102,146 105 likes Comments 0 Shares: 75 Reward $12.50 Status Pending")
    assert fields["views"] == 102146
    assert fields["likes"] == 105
    assert fields["comments"] == 0
    assert fields["shares"] == 75
    assert fields["payout"] == 12.5
    assert fields["status"].lower().startswith("pending")


def test_ocr_text_metric_extraction_handles_dot_thousands_separator() -> None:
    fields = extract_visible_metrics("Views 102.146 Likes 105 Comments 0 Shares 75")
    assert fields["views"] == 102146
    fields = extract_visible_metrics("Views 1.348.911 Likes 184.982 Comments 1.131 Shares 6.411")
    assert fields["views"] == 1348911
    assert fields["likes"] == 184982
    assert fields["comments"] == 1131
    assert fields["shares"] == 6411


def test_ocr_text_metric_extraction_does_not_reuse_previous_metric_value() -> None:
    fields = extract_visible_metrics("Views\n102.148\nLikes\n\n105\nComments\n°\nshares\n\ni\n")
    assert fields["views"] == 102148
    assert fields["likes"] == 105
    assert fields["comments"] is None


def test_ocr_text_metric_extraction_supports_whop_stacked_layout() -> None:
    fields = extract_visible_metrics("Views\n102,146\nLikes\n105\nComments\n0\nShares\n75\nApprove")
    assert fields["views"] == 102146
    assert fields["likes"] == 105
    assert fields["comments"] == 0
    assert fields["shares"] == 75
    assert fields["status"] == "Approve visible"


def test_strict_moderation_rejects_high_views_zero_comments() -> None:
    prediction = predict_one(
        {
            "platform": "TikTok",
            "views": 102146,
            "likes": 105,
            "comments": 0,
            "shares": 75,
            "graph_pattern": "flatline_then_vertical_spike",
            "avg_views_30d": 2176,
        }
    )
    moderation = prediction["moderation"]
    assert moderation["decision"] == "Reject: Send Full Analytics"
    assert moderation["confidence"] in {"High", "Medium"}
    assert any("zero comments" in item.lower() for item in moderation["red_flags"])


def test_strict_moderation_rejects_missing_screenshot_metrics() -> None:
    prediction = predict_one({"platform": "TikTok", "views": 0, "likes": 0, "comments": 0, "shares": 0})
    analysis = {
        "fields": {"views": None, "likes": None, "comments": None, "shares": None},
        "graph_shape": "unknown",
        "error": "OCR failed",
    }
    moderation = moderation_decision_from_prediction(
        prediction,
        screenshot_analysis=analysis,
        submission={"platform": "TikTok", "views": 0, "likes": 0, "comments": 0, "shares": 0},
    )
    assert moderation["decision"] == "Reject: Send Full Analytics"
    assert any("missing visible metric" in item.lower() for item in moderation["red_flags"])


def test_strict_moderation_can_approve_clean_proportional_submission() -> None:
    prediction = predict_one(
        {
            "platform": "TikTok",
            "views": 50000,
            "likes": 3500,
            "comments": 220,
            "shares": 300,
            "graph_pattern": "smooth_gradual",
            "completion_rate": 0.55,
            "profile_traffic_pct": 0.10,
            "tier1_audience_pct": 0.72,
            "creator_followers": 25000,
            "avg_views_30d": 12000,
        }
    )
    moderation = prediction["moderation"]
    assert moderation["decision"] == "Approved"
    assert moderation["red_flags"] == ["No major red flags found."]


def test_screenshot_analysis_converts_fields_to_submission() -> None:
    analysis = {
        "fields": {"platform": "TikTok", "views": 102146, "likes": 105, "comments": 0, "shares": 75},
        "graph_shape": "flatline_then_vertical_spike",
    }
    submission = extracted_metrics_to_submission(analysis)
    assert submission["platform"] == "TikTok"
    assert submission["views"] == 102146
    assert submission["graph_pattern"] == "flatline_then_vertical_spike"


def test_root_streamlit_app_rerenders_after_screenshot_upload() -> None:
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)
    app.run()
    app.file_uploader(key="screenshot_upload").upload("sample.png", _png_bytes(), "image/png")
    app.run(timeout=30)

    assert not app.exception
    assert any(title.value == "AI Video Fraud Review" for title in app.title)
    assert any(success.value == "File ready: sample.png" for success in app.success)
    assert any(button.label == "Analyze Uploaded Screenshot" for button in app.button)


def test_screenshot_override_inputs_reset_for_new_upload() -> None:
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)
    app.run()
    app.file_uploader(key="screenshot_upload").upload("first.png", _png_bytes("#222222"), "image/png")
    app.run(timeout=30)
    app.text_input[0].input("102146")
    app.run(timeout=30)

    app.file_uploader(key="screenshot_upload").upload("second.png", _png_bytes("#333333"), "image/png")
    app.run(timeout=30)

    assert not app.exception
    assert [text_input.value for text_input in app.text_input[:4]] == ["", "", "", ""]


def test_metric_overrides_replace_missing_ocr_fields() -> None:
    analysis = {
        "fields": {"platform": "TikTok", "views": None, "likes": None, "comments": None, "shares": None},
        "missing_fields": ["views", "likes", "comments", "shares"],
    }
    updated = apply_metric_overrides(
        analysis,
        {"views": "102,146", "likes": "105", "comments": "0", "shares": "75"},
    )

    assert updated["fields"]["views"] == 102146
    assert updated["fields"]["comments"] == 0
    assert "views" not in updated["missing_fields"]
    assert updated["field_sources"]["shares"] == "manual_override"
