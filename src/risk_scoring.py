"""Rule and model based risk scoring."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .utils import clamp, decision_from_score, risk_level_from_score, safe_divide


CLASS_RISK_WEIGHTS = {
    "clean": 8,
    "suspicious": 62,
    "botted": 94,
}


def _value(row: dict[str, Any] | pd.Series, name: str, default: Any = 0) -> Any:
    if isinstance(row, pd.Series):
        return row.get(name, default)
    return row.get(name, default)


def calculate_rule_score(row: dict[str, Any] | pd.Series) -> int:
    views = float(_value(row, "views", 0) or 0)
    comments = float(_value(row, "comments", 0) or 0)
    like_rate = float(_value(row, "like_rate", 0) or 0)
    comment_rate = float(_value(row, "comment_rate", 0) or 0)
    total_engagement_rate = float(_value(row, "total_engagement_rate", _value(row, "engagement_rate", 0)) or 0)
    completion_rate = float(_value(row, "completion_rate", 0) or 0)
    avg_watch_time = float(_value(row, "avg_watch_time_sec", _value(row, "watch_time", 0)) or 0)
    video_length = float(_value(row, "video_length_sec", 0) or 0)
    watch_ratio = safe_divide(avg_watch_time, video_length, 0)

    ratio_score = float(_value(row, "suspicious_ratio_score", 0) or 0)
    graph_score = float(_value(row, "graph_suspicion_score", 0) or 0)
    alignment_score = float(_value(row, "engagement_alignment_score", 50) or 50)
    view_jump_ratio = float(_value(row, "view_jump_ratio", 0) or 0)
    robust_spike_ratio = float(_value(row, "robust_spike_ratio", 0) or 0)
    max_jump_zscore = float(_value(row, "max_view_jump_zscore", 0) or 0)
    top_two_view_share = float(_value(row, "top_two_view_jump_share", 0) or 0)
    vertical_spike = float(_value(row, "vertical_spike_flag", 0) or 0)
    flatline_ratio = float(_value(row, "flatline_ratio", 0) or 0)
    pre_spike_flatline = float(_value(row, "pre_spike_flatline_hours", 0) or 0)
    flatline_then_spike_score = float(_value(row, "flatline_then_spike_score", 0) or 0)
    spike_after_flatline = float(_value(row, "spike_after_flatline", 0) or 0)
    divergence = float(_value(row, "views_engagement_divergence", 0) or 0)
    spike_gap_ratio = float(_value(row, "spike_engagement_gap_ratio", 0) or 0)
    engagement_lag_score = float(_value(row, "engagement_lag_score", 0) or 0)
    late_like_spike = float(_value(row, "late_like_spike", 0) or 0)
    late_like_strength = float(_value(row, "late_like_spike_strength", 0) or 0)
    view_like_lag = float(_value(row, "view_like_peak_lag_hours", 0) or 0)

    comment_quality = str(_value(row, "comment_quality", "")).lower()
    suspicious_comment_pct = float(_value(row, "suspicious_comment_pct", 0) or 0)
    repeated_comment_pct = float(_value(row, "repeated_comment_pct", 0) or 0)
    bot_like_profile_pct = float(_value(row, "bot_like_profile_pct", 0) or 0)
    views_to_avg_multiplier = float(_value(row, "views_to_avg_multiplier", 0) or 0)

    score = ratio_score * 0.34 + graph_score * 0.38 + max(0, 100 - alignment_score) * 0.14

    if views >= 10000 and comments == 0:
        score += 10
    if views >= 100000 and comments <= 3:
        score += 8
    if views >= 10000 and like_rate < 0.002:
        score += 7
    if total_engagement_rate < 0.005 and views >= 10000:
        score += 7
    if completion_rate and completion_rate < 0.08:
        score += 5
    if watch_ratio and watch_ratio < 0.10:
        score += 5
    if suspicious_comment_pct >= 0.50:
        score += 5
    if repeated_comment_pct >= 0.35:
        score += 4
    if bot_like_profile_pct >= 0.45:
        score += 5
    if views_to_avg_multiplier >= 25:
        score += 6
    if vertical_spike or view_jump_ratio >= 0.45 or robust_spike_ratio >= 12 or max_jump_zscore >= 3:
        score += 10
    if top_two_view_share >= 0.60:
        score += 6
    if spike_after_flatline or flatline_then_spike_score >= 55 or (pre_spike_flatline >= 4 and view_jump_ratio >= 0.20):
        score += 10
    if flatline_ratio >= 0.30:
        score += 5
    if divergence >= 0.25:
        score += 7
    if spike_gap_ratio >= 0.60 or engagement_lag_score >= 60:
        score += 7
    if late_like_spike or (late_like_strength >= 0.25 and view_like_lag >= 4):
        score += 6
    if any(token in comment_quality for token in ["emoji_spam", "generic", "unrelated"]):
        score += 4
    if str(_value(row, "graph_pattern", "")).lower() in {"smooth_gradual", "normal_viral_smooth"} and total_engagement_rate >= 0.02:
        score -= 12

    return int(round(clamp(score)))


def calculate_risk_score(
    row: dict[str, Any] | pd.Series,
    model_probabilities: dict[str, float] | None = None,
) -> int:
    rule_score = calculate_rule_score(row)
    if not model_probabilities:
        return rule_score

    model_score = 0.0
    for label, probability in model_probabilities.items():
        model_score += CLASS_RISK_WEIGHTS.get(str(label).lower(), 50) * float(probability)

    combined = rule_score * 0.58 + model_score * 0.42
    if rule_score >= 85:
        combined = max(combined, rule_score - 3)
    if rule_score <= 20 and model_score <= 30:
        combined = min(combined, 24)
    return int(round(clamp(combined)))


def class_from_score(score: int) -> str:
    if score >= 75:
        return "Botted"
    if score >= 25:
        return "Suspicious"
    return "Clean"


def fraud_type_levels(row: dict[str, Any] | pd.Series, score: int) -> dict[str, str]:
    def level(points: float) -> str:
        if points >= 70:
            return "High"
        if points >= 35:
            return "Medium"
        return "Low"

    views = float(_value(row, "views", 0) or 0)
    comments = float(_value(row, "comments", 0) or 0)
    comment_quality = str(_value(row, "comment_quality", "")).lower()
    graph_pattern = str(_value(row, "graph_pattern", "")).lower()
    ratio_score = float(_value(row, "suspicious_ratio_score", 0) or 0)
    graph_score = float(_value(row, "graph_suspicion_score", 0) or 0)
    view_jump_ratio = float(_value(row, "view_jump_ratio", 0) or 0)
    vertical_spike = float(_value(row, "vertical_spike_flag", 0) or 0)
    flatline_then_spike_score = float(_value(row, "flatline_then_spike_score", 0) or 0)
    spike_after_flatline = float(_value(row, "spike_after_flatline", 0) or 0)
    divergence = float(_value(row, "views_engagement_divergence", 0) or 0)
    spike_gap_ratio = float(_value(row, "spike_engagement_gap_ratio", 0) or 0)
    engagement_lag_score = float(_value(row, "engagement_lag_score", 0) or 0)
    late_like_spike = float(_value(row, "late_like_spike", 0) or 0)
    late_like_strength = float(_value(row, "late_like_spike_strength", 0) or 0)
    repeated_comment_pct = float(_value(row, "repeated_comment_pct", 0) or 0)
    suspicious_comment_pct = float(_value(row, "suspicious_comment_pct", 0) or 0)
    bot_like_profile_pct = float(_value(row, "bot_like_profile_pct", 0) or 0)
    views_to_avg_multiplier = float(_value(row, "views_to_avg_multiplier", 0) or 0)
    completion_rate = float(_value(row, "completion_rate", 0) or 0)

    fake_views = max(ratio_score, graph_score, view_jump_ratio * 100, divergence * 100, spike_gap_ratio * 100)
    fake_likes = max(bot_like_profile_pct * 100, late_like_strength * 100, 65 if late_like_spike else 0)
    fake_comments = max(suspicious_comment_pct * 100, repeated_comment_pct * 100)
    if comments == 0:
        fake_comments = min(fake_comments, 25)
    fake_boost = max(graph_score, flatline_then_spike_score, min(100, views_to_avg_multiplier * 2))
    autoclick_patterns = {
        "vertical_spike_no_engagement",
        "flatline_then_vertical_spike",
        "flat_then_spike",
        "step_like_batches",
    }
    autoclick = max(
        graph_score if graph_pattern in autoclick_patterns else 0,
        80 if vertical_spike else 0,
        75 if spike_after_flatline else 0,
        engagement_lag_score,
        70 if completion_rate and completion_rate < 0.08 else 0,
    )
    engagement_pod = max(repeated_comment_pct * 100, bot_like_profile_pct * 85, 70 if "generic" in comment_quality else 0)

    if score >= 75:
        fake_views = max(fake_views, 70)
    return {
        "fake_views": level(fake_views),
        "fake_likes": level(fake_likes),
        "fake_comments": level(fake_comments),
        "fake_boost": level(fake_boost),
        "autoclick_autoswipe_bot": level(autoclick),
        "engagement_pod": level(engagement_pod),
    }


def recommendation(score: int, row: dict[str, Any] | pd.Series | None = None) -> str:
    return decision_from_score(score, row)


def risk_level(score: int) -> str:
    return risk_level_from_score(score)
