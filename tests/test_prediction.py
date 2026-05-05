from __future__ import annotations

import pandas as pd

from src.feature_engineering import build_feature_frame, create_graph_features
from src.predict import predict_one
from src.risk_scoring import calculate_risk_score, risk_level
from src.screenshot_analyzer import extract_visible_metrics


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
