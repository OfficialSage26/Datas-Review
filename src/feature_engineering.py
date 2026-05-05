"""Feature engineering for video engagement fraud detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import clamp, normalize_columns, safe_divide


BASE_NUMERIC_COLUMNS = [
    "views",
    "likes",
    "comments",
    "shares",
    "saves",
    "reach",
    "avg_views_30d",
    "avg_watch_time_sec",
    "watch_time",
    "completion_rate",
    "max_view_jump_pct",
    "flatline_hours_before_jump",
    "suspicious_comment_pct",
    "repeated_comment_pct",
    "bot_like_profile_pct",
    "views_to_avg_multiplier",
]

RED_GRAPH_PATTERNS = {
    "vertical_spike_no_engagement": 95,
    "flatline_then_vertical_spike": 92,
    "flat_then_spike": 82,
    "step_like_batches": 72,
    "likes_freeze": 78,
    "late_like_spike": 68,
    "low_engagement_smooth": 45,
    "normal_viral_smooth": 8,
    "smooth_gradual": 5,
}


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].fillna(False).astype(bool)


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    result = normalize_columns(df)
    for column in BASE_NUMERIC_COLUMNS:
        if column not in result.columns:
            result[column] = 0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)

    views = result["views"].replace(0, np.nan)
    result["like_rate"] = (result["likes"] / views).replace([np.inf, -np.inf], np.nan).fillna(0)
    result["comment_rate"] = (result["comments"] / views).replace([np.inf, -np.inf], np.nan).fillna(0)
    result["share_rate"] = (result["shares"] / views).replace([np.inf, -np.inf], np.nan).fillna(0)
    result["total_engagement_rate"] = (
        (result["likes"] + result["comments"] + result["shares"]) / views
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    result["engagement_rate"] = result["total_engagement_rate"]
    result["reach_view_ratio"] = (result["reach"] / views).replace([np.inf, -np.inf], np.nan).fillna(0)
    result["views_to_avg_multiplier"] = (
        result["views"] / result["avg_views_30d"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    return result


def _alignment(view_delta: np.ndarray, metric_delta: np.ndarray) -> float:
    if len(view_delta) < 2 or np.std(view_delta) == 0 or np.std(metric_delta) == 0:
        return 0.0
    corr = float(np.corrcoef(view_delta, metric_delta)[0, 1])
    if np.isnan(corr) or np.isinf(corr):
        return 0.0
    return clamp((corr + 1) * 50, 0, 100)


def _longest_flatline_hours(view_delta: np.ndarray, hour_delta: np.ndarray, final_views: float) -> float:
    threshold = max(1.0, final_views * 0.001)
    longest = 0.0
    current = 0.0
    for delta, hours in zip(view_delta, hour_delta):
        if delta <= threshold:
            current += max(float(hours), 0.0)
            longest = max(longest, current)
        else:
            current = 0.0
    return longest


def _spike_after_flatline(view_delta: np.ndarray, hour_delta: np.ndarray, final_views: float) -> int:
    threshold = max(10.0, final_views * 0.25)
    flat_threshold = max(1.0, final_views * 0.001)
    current_flat_hours = 0.0
    for delta, hours in zip(view_delta, hour_delta):
        if delta <= flat_threshold:
            current_flat_hours += max(float(hours), 0.0)
        elif delta >= threshold and current_flat_hours >= 4:
            return 1
        else:
            current_flat_hours = 0.0
    return 0


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.all(values == 0):
        return 0.0
    values = np.sort(np.maximum(values, 0))
    n = len(values)
    weighted = np.sum((np.arange(1, n + 1) * values))
    total = np.sum(values)
    if total == 0:
        return 0.0
    return float((2 * weighted) / (n * total) - (n + 1) / n)


def _pre_spike_flatline_hours(
    view_delta: np.ndarray,
    hour_delta: np.ndarray,
    final_views: float,
    spike_index: int,
) -> float:
    flat_threshold = max(1.0, final_views * 0.0015)
    total = 0.0
    for index in range(spike_index - 1, -1, -1):
        if view_delta[index] <= flat_threshold:
            total += max(float(hour_delta[index]), 0.0)
        else:
            break
    return total


def _late_metric_spike(
    view_delta: np.ndarray,
    metric_delta: np.ndarray,
    hours: np.ndarray,
    final_metric: float,
) -> tuple[int, float, float]:
    if len(metric_delta) == 0 or final_metric <= 0:
        return 0, 0.0, 0.0
    view_peak_index = int(view_delta.argmax()) if len(view_delta) else 0
    metric_peak_index = int(metric_delta.argmax())
    metric_share = safe_divide(float(metric_delta[metric_peak_index]), final_metric, 0.0)
    lag_hours = float(max(0.0, hours[min(metric_peak_index + 1, len(hours) - 1)] - hours[min(view_peak_index + 1, len(hours) - 1)]))
    is_late = int(metric_peak_index > view_peak_index and lag_hours >= 4 and metric_share >= 0.20)
    return is_late, metric_share, lag_hours


def create_graph_features(graph_df: pd.DataFrame | None) -> pd.DataFrame:
    if graph_df is None or graph_df.empty:
        return pd.DataFrame()

    graph = normalize_columns(graph_df)
    required = ["submission_id", "hour_since_post", "views_cumulative"]
    if any(column not in graph.columns for column in required):
        return pd.DataFrame()

    for column in [
        "hour_since_post",
        "views_cumulative",
        "likes_cumulative",
        "comments_cumulative",
        "shares_cumulative",
    ]:
        if column not in graph.columns:
            graph[column] = 0
        graph[column] = pd.to_numeric(graph[column], errors="coerce").fillna(0)

    records: list[dict[str, Any]] = []
    for submission_id, group in graph.sort_values(["submission_id", "hour_since_post"]).groupby("submission_id"):
        hours = group["hour_since_post"].to_numpy(dtype=float)
        views = group["views_cumulative"].to_numpy(dtype=float)
        likes = group["likes_cumulative"].to_numpy(dtype=float)
        comments = group["comments_cumulative"].to_numpy(dtype=float)
        shares = group["shares_cumulative"].to_numpy(dtype=float)

        if len(views) < 2:
            continue

        hour_delta = np.diff(hours)
        hour_delta = np.where(hour_delta <= 0, 1.0, hour_delta)
        view_delta = np.maximum(np.diff(views), 0)
        like_delta = np.maximum(np.diff(likes), 0)
        comment_delta = np.maximum(np.diff(comments), 0)
        share_delta = np.maximum(np.diff(shares), 0)
        engagement_delta = like_delta + comment_delta + share_delta

        final_views = float(max(views[-1], views.max(), 0))
        final_likes = float(max(likes[-1], likes.max(), 0))
        final_engagement = float(max(final_likes + comments[-1] + shares[-1], 0))
        max_jump = float(view_delta.max()) if len(view_delta) else 0.0
        max_jump_index = int(view_delta.argmax()) if len(view_delta) else 0
        mean_growth = float((view_delta / hour_delta).mean()) if len(view_delta) else 0.0
        median_growth = float(np.median(view_delta / hour_delta)) if len(view_delta) else 0.0
        growth_volatility = safe_divide(float(view_delta.std()), float(view_delta.mean()), 0.0)
        robust_spike_ratio = safe_divide(max_jump, max(float(np.median(view_delta)), 1.0), 0.0)
        max_jump_zscore = safe_divide(max_jump - float(view_delta.mean()), float(view_delta.std()), 0.0)
        top_two_view_share = safe_divide(float(np.sort(view_delta)[-2:].sum()), final_views, 0.0)
        graph_irregularity_score = clamp(_gini(view_delta) * 100, 0, 100)
        spike_threshold = max(final_views * 0.10, float(view_delta.mean() + 1.5 * view_delta.std()))
        number_of_spikes = int((view_delta >= spike_threshold).sum()) if final_views > 0 else 0
        vertical_spike_flag = int(final_views > 0 and (max_jump / final_views >= 0.45 or robust_spike_ratio >= 12 or max_jump_zscore >= 3))
        high_view_threshold = max(final_views * 0.05, float(view_delta.mean()))
        divergence_mask = (view_delta > high_view_threshold) & (engagement_delta <= max(1.0, final_engagement * 0.02))
        divergence = safe_divide(int(divergence_mask.sum()), len(view_delta), 0.0)
        spike_engagement = float(engagement_delta[max_jump_index]) if len(engagement_delta) else 0.0
        spike_engagement_rate = safe_divide(spike_engagement, max_jump, 0.0)
        final_engagement_rate = safe_divide(final_engagement, final_views, 0.0)
        spike_engagement_gap = max(0.0, final_engagement_rate - spike_engagement_rate)
        spike_engagement_gap_ratio = safe_divide(spike_engagement_gap, max(final_engagement_rate, 0.000001), 0.0)
        view_engagement_correlation = 0.0
        if len(view_delta) >= 2 and np.std(view_delta) > 0 and np.std(engagement_delta) > 0:
            view_engagement_correlation = float(np.corrcoef(view_delta, engagement_delta)[0, 1])
            if np.isnan(view_engagement_correlation) or np.isinf(view_engagement_correlation):
                view_engagement_correlation = 0.0
        engagement_lag_score = clamp((1 - max(view_engagement_correlation, -1)) * 35 + spike_engagement_gap_ratio * 65, 0, 100)
        like_freeze = int(((view_delta > high_view_threshold) & (like_delta <= 0)).any())
        late_like_spike, late_like_spike_strength, view_like_peak_lag_hours = _late_metric_spike(
            view_delta,
            like_delta,
            hours,
            final_likes,
        )
        pre_spike_flatline = _pre_spike_flatline_hours(view_delta, hour_delta, final_views, max_jump_index)
        longest_flatline = _longest_flatline_hours(view_delta, hour_delta, final_views)
        flatline_ratio = safe_divide(longest_flatline, float(max(hours[-1] - hours[0], 1)), 0.0)
        flatline_then_spike_score = clamp(
            safe_divide(pre_spike_flatline, 8, 0.0) * 35
            + safe_divide(max_jump, final_views, 0.0) * 65,
            0,
            100,
        )

        records.append(
            {
                "submission_id": submission_id,
                "max_view_jump": max_jump,
                "view_jump_ratio": safe_divide(max_jump, final_views, 0.0),
                "max_view_jump_zscore": max_jump_zscore,
                "robust_spike_ratio": robust_spike_ratio,
                "top_two_view_jump_share": top_two_view_share,
                "vertical_spike_flag": vertical_spike_flag,
                "number_of_spikes": number_of_spikes,
                "average_growth_rate": mean_growth,
                "median_growth_rate": median_growth,
                "growth_volatility": growth_volatility,
                "graph_irregularity_score": graph_irregularity_score,
                "flatline_duration": longest_flatline,
                "flatline_ratio": flatline_ratio,
                "pre_spike_flatline_hours": pre_spike_flatline,
                "flatline_then_spike_score": flatline_then_spike_score,
                "spike_after_flatline": _spike_after_flatline(view_delta, hour_delta, final_views),
                "step_like_growth": int(number_of_spikes >= 3 or growth_volatility >= 1.25 or top_two_view_share >= 0.55),
                "views_engagement_divergence": divergence,
                "spike_engagement_rate": spike_engagement_rate,
                "spike_engagement_gap_ratio": spike_engagement_gap_ratio,
                "view_engagement_correlation": view_engagement_correlation,
                "engagement_lag_score": engagement_lag_score,
                "likes_freeze_during_view_growth": like_freeze,
                "late_like_spike": late_like_spike,
                "late_like_spike_strength": late_like_spike_strength,
                "view_like_peak_lag_hours": view_like_peak_lag_hours,
                "comment_growth_alignment": _alignment(view_delta, comment_delta),
                "share_growth_alignment": _alignment(view_delta, share_delta),
            }
        )

    return pd.DataFrame(records)


def add_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    result = add_ratio_features(df)
    views = _numeric_series(result, "views")
    comments = _numeric_series(result, "comments")
    shares = _numeric_series(result, "shares")
    like_rate = _numeric_series(result, "like_rate")
    comment_rate = _numeric_series(result, "comment_rate")
    share_rate = _numeric_series(result, "share_rate")

    result["high_views_low_likes"] = ((views >= 10000) & (like_rate < 0.005)).astype(int)
    result["high_views_low_comments"] = ((views >= 10000) & ((comments <= 3) | (comment_rate < 0.0002))).astype(int)
    result["high_views_zero_comments"] = ((views >= 10000) & (comments == 0)).astype(int)
    result["high_views_low_shares"] = ((views >= 10000) & ((shares <= 3) | (share_rate < 0.0001))).astype(int)

    result["suspicious_ratio_score"] = (
        result["high_views_low_likes"] * 28
        + result["high_views_low_comments"] * 28
        + result["high_views_zero_comments"] * 18
        + result["high_views_low_shares"] * 16
        + (result["total_engagement_rate"] < 0.005).astype(int) * 10
    ).clip(0, 100)

    graph_pattern = result.get("graph_pattern", pd.Series("unknown", index=result.index)).fillna("unknown").astype(str).str.lower()
    pattern_score = graph_pattern.map(RED_GRAPH_PATTERNS).fillna(20).astype(float)

    max_jump_pct = _numeric_series(result, "max_view_jump_pct")
    view_jump_ratio = _numeric_series(result, "view_jump_ratio")
    max_jump_zscore = _numeric_series(result, "max_view_jump_zscore")
    robust_spike_ratio = _numeric_series(result, "robust_spike_ratio")
    top_two_share = _numeric_series(result, "top_two_view_jump_share")
    vertical_spike = _numeric_series(result, "vertical_spike_flag")
    flatline = _numeric_series(result, "flatline_duration")
    flatline_ratio = _numeric_series(result, "flatline_ratio")
    pre_spike_flatline = _numeric_series(result, "pre_spike_flatline_hours")
    flatline_then_spike_score = _numeric_series(result, "flatline_then_spike_score")
    flatline_hours_before_jump = _numeric_series(result, "flatline_hours_before_jump")
    spike_after_flatline = _numeric_series(result, "spike_after_flatline")
    step_like_growth = _numeric_series(result, "step_like_growth")
    divergence = _numeric_series(result, "views_engagement_divergence")
    spike_gap_ratio = _numeric_series(result, "spike_engagement_gap_ratio")
    engagement_lag_score = _numeric_series(result, "engagement_lag_score")
    graph_irregularity = _numeric_series(result, "graph_irregularity_score")
    like_freeze = _bool_series(result, "like_freeze_flag").astype(int) | _numeric_series(result, "likes_freeze_during_view_growth").astype(int)
    late_like = _bool_series(result, "late_like_spike_flag").astype(int) | _numeric_series(result, "late_like_spike").astype(int)
    late_like_strength = _numeric_series(result, "late_like_spike_strength")
    view_like_lag = _numeric_series(result, "view_like_peak_lag_hours")
    no_matching = _bool_series(result, "no_matching_engagement_flag").astype(int)

    graph_suspicion_score = (
        pattern_score
        + (max_jump_pct >= 1.0).astype(int) * 12
        + (view_jump_ratio >= 0.35).astype(int) * 16
        + (max_jump_zscore >= 3).astype(int) * 10
        + (robust_spike_ratio >= 10).astype(int) * 10
        + (top_two_share >= 0.55).astype(int) * 8
        + vertical_spike.astype(int) * 12
        + ((flatline >= 4) | (flatline_hours_before_jump >= 8)).astype(int) * 12
        + ((flatline_ratio >= 0.25) | (pre_spike_flatline >= 4)).astype(int) * 10
        + (flatline_then_spike_score * 0.20)
        + spike_after_flatline.astype(int) * 15
        + step_like_growth.astype(int) * 10
        + like_freeze.astype(int) * 14
        + late_like.astype(int) * 10
        + (late_like_strength >= 0.25).astype(int) * 8
        + (view_like_lag >= 4).astype(int) * 6
        + no_matching.astype(int) * 12
        + (divergence * 25)
        + (spike_gap_ratio * 18)
        + (engagement_lag_score * 0.18)
        + (graph_irregularity * 0.10)
    ).clip(0, 100)
    healthy_graph_pattern = graph_pattern.isin({"smooth_gradual", "normal_viral_smooth"})
    hard_graph_evidence = (
        vertical_spike.astype(bool)
        | spike_after_flatline.astype(bool)
        | like_freeze.astype(bool)
        | late_like.astype(bool)
        | no_matching.astype(bool)
        | ((pre_spike_flatline >= 4) & (view_jump_ratio >= 0.25))
        | (view_jump_ratio >= 0.45)
        | (robust_spike_ratio >= 12)
    )
    healthy_graph_cap = healthy_graph_pattern & (result["total_engagement_rate"] >= 0.02) & ~hard_graph_evidence
    result["graph_suspicion_score"] = graph_suspicion_score.where(
        ~healthy_graph_cap,
        graph_suspicion_score.clip(upper=45),
    )

    comment_alignment = _numeric_series(result, "comment_growth_alignment", 50.0)
    share_alignment = _numeric_series(result, "share_growth_alignment", 50.0)
    result["engagement_alignment_score"] = (
        (comment_alignment * 0.45 + share_alignment * 0.35 + (100 - divergence * 100) * 0.20)
        - like_freeze.astype(int) * 12
        - no_matching.astype(int) * 15
        - (spike_gap_ratio * 20)
        - (engagement_lag_score * 0.15)
    ).clip(0, 100)

    return result


def build_feature_frame(submissions: pd.DataFrame, graph_df: pd.DataFrame | None = None) -> pd.DataFrame:
    result = add_ratio_features(submissions)
    if "submission_id" not in result.columns:
        result["submission_id"] = [f"INPUT-{idx:06d}" for idx in range(len(result))]

    graph_features = create_graph_features(graph_df)
    if not graph_features.empty and "submission_id" in result.columns:
        result = result.merge(graph_features, on="submission_id", how="left")

    result = add_risk_flags(result)

    numeric_feature_columns = [
        "max_view_jump",
        "view_jump_ratio",
        "max_view_jump_zscore",
        "robust_spike_ratio",
        "top_two_view_jump_share",
        "vertical_spike_flag",
        "number_of_spikes",
        "average_growth_rate",
        "median_growth_rate",
        "growth_volatility",
        "graph_irregularity_score",
        "flatline_duration",
        "flatline_ratio",
        "pre_spike_flatline_hours",
        "flatline_then_spike_score",
        "spike_after_flatline",
        "step_like_growth",
        "views_engagement_divergence",
        "spike_engagement_rate",
        "spike_engagement_gap_ratio",
        "view_engagement_correlation",
        "engagement_lag_score",
        "likes_freeze_during_view_growth",
        "late_like_spike",
        "late_like_spike_strength",
        "view_like_peak_lag_hours",
        "comment_growth_alignment",
        "share_growth_alignment",
        "suspicious_ratio_score",
        "graph_suspicion_score",
        "engagement_alignment_score",
    ]
    for column in numeric_feature_columns:
        if column not in result.columns:
            result[column] = 0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)

    return result
