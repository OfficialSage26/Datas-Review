"""Generate supplemental synthetic training data.

The generated rows are for model prototyping only. They do not replace real
reviewed submissions and are written to separate files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import (
    GENERATED_GRAPH_CSV,
    GENERATED_SUBMISSION_CSV,
    GRAPH_CSV,
    SUBMISSION_CSV,
)
from .utils import DATAS_DIR, clamp, safe_divide


HOURS = np.arange(0, 25, 2)
LABELS = ["clean", "suspicious", "botted"]
PLATFORMS = ["TikTok", "Instagram", "YouTube Shorts"]
CAMPAIGNS = [
    "Car Slideshows",
    "Tech Gadget",
    "Anime Edit",
    "Mobile Game Speedrun",
    "Finance Explainer",
    "Fitness Clip",
    "Travel Hook",
    "AI App Demo",
]


def _choice(rng: np.random.Generator, values: list[Any], probabilities: list[float] | None = None) -> Any:
    return values[int(rng.choice(len(values), p=probabilities))]


def _bounded_lognormal(rng: np.random.Generator, median: float, sigma: float, low: int, high: int) -> int:
    value = int(round(rng.lognormal(np.log(max(median, 1)), sigma)))
    return int(clamp(value, low, high))


def _increments(final_value: int, weights: np.ndarray) -> np.ndarray:
    final_value = max(int(final_value), 0)
    if final_value == 0:
        return np.zeros(len(weights), dtype=int)
    weights = np.maximum(weights.astype(float), 0)
    if weights.sum() == 0:
        weights = np.ones(len(weights), dtype=float)
    weights = weights / weights.sum()
    raw = weights * final_value
    increments = np.floor(raw).astype(int)
    remainder = final_value - int(increments.sum())
    if remainder > 0:
        order = np.argsort(raw - increments)[::-1]
        for index in order[:remainder]:
            increments[index] += 1
    return increments


def _weights_for_pattern(
    rng: np.random.Generator,
    pattern: str,
    metric: str = "views",
    label: str = "clean",
) -> np.ndarray:
    n = len(HOURS) - 1
    weights = rng.gamma(2.0, 1.0, size=n)

    if pattern in {"smooth_gradual", "normal_viral_smooth"}:
        trend = np.linspace(0.7, 1.3, n)
        return weights * trend

    if pattern == "low_engagement_smooth":
        return weights * np.linspace(0.8, 1.2, n)

    if pattern == "vertical_spike_no_engagement":
        weights = np.ones(n) * 0.02
        spike = int(rng.integers(3, 9))
        weights[spike] = 0.90
        return weights

    if pattern == "flat_then_spike":
        weights = np.ones(n) * 0.02
        flat_until = int(rng.integers(3, 7))
        weights[:flat_until] = 0.002
        weights[flat_until] = 0.72
        weights[flat_until + 1 :] += rng.random(n - flat_until - 1) * 0.08
        return weights

    if pattern == "step_like_batches":
        weights = np.ones(n) * 0.01
        pulses = rng.choice(np.arange(1, n), size=int(rng.integers(3, 6)), replace=False)
        weights[pulses] = rng.uniform(0.18, 0.40, size=len(pulses))
        return weights

    if pattern == "likes_freeze" and metric == "likes":
        weights = np.zeros(n)
        cutoff = int(rng.integers(3, 6))
        weights[:cutoff] = rng.gamma(2.0, 1.0, size=cutoff)
        return weights

    if pattern == "late_like_spike" and metric == "likes":
        weights = np.ones(n) * 0.02
        spike = int(rng.integers(8, n))
        weights[spike] = 0.80
        return weights

    if label in {"suspicious", "botted"} and metric in {"comments", "shares"}:
        weights = np.ones(n) * 0.02
        if rng.random() < 0.6:
            weights[: int(rng.integers(5, 9))] = 0
        return weights

    return weights


def _cumulative(final_value: int, weights: np.ndarray) -> list[int]:
    increments = _increments(final_value, weights)
    return [0] + np.cumsum(increments).astype(int).tolist()


def _risk_metadata(label: str, rng: np.random.Generator) -> tuple[int, str, str]:
    if label == "clean":
        score = int(rng.integers(5, 24))
        return score, "Low", "Approve / monitor later"
    if label == "suspicious":
        score = int(rng.integers(38, 75))
        level = "Medium" if score <= 49 else "High"
        decision = "Needs Manual Review" if score <= 49 else "Reject and ask for analytics in ticket"
        return score, level, decision
    score = int(rng.integers(78, 100))
    return score, "Critical", "Reject"


def _fraud_type(label: str, pattern: str, rng: np.random.Generator) -> str:
    if label == "clean":
        return "none"
    if pattern == "likes_freeze":
        return _choice(rng, ["fake_likes", "possible_engagement_pod", "possible_fake_views"])
    if pattern == "late_like_spike":
        return _choice(rng, ["fake_likes", "possible_fake_boost", "possible_engagement_pod"])
    if pattern == "step_like_batches":
        return _choice(rng, ["autoclick_autoswipe", "possible_autoswipe", "view_injection"])
    if pattern == "flat_then_spike":
        return _choice(rng, ["fake_boost", "needs_analytics", "possible_fake_boost"])
    if pattern == "vertical_spike_no_engagement":
        return _choice(rng, ["fake_views", "view_injection", "autoclick_autoswipe"])
    return _choice(rng, ["possible_fake_views", "needs_analytics"])


def _row_for_label(index: int, label: str, rng: np.random.Generator) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    platform = _choice(rng, PLATFORMS)
    campaign = _choice(rng, CAMPAIGNS)
    days_live = int(rng.integers(1, 15))
    followers = _bounded_lognormal(rng, 6500 if label != "clean" else 18000, 1.7, 50, 500000)
    video_length = int(_choice(rng, [12, 15, 20, 30, 45, 60]))

    if label == "clean":
        views = _bounded_lognormal(rng, 45000, 1.15, 150, 2_500_000)
        avg_multiplier = float(rng.uniform(0.8, 7.5))
        like_rate = float(rng.uniform(0.025, 0.12))
        comment_rate = float(rng.uniform(0.0007, 0.012))
        share_rate = float(rng.uniform(0.0005, 0.010))
        pattern = _choice(rng, ["smooth_gradual", "normal_viral_smooth"], [0.58, 0.42])
        retention = _choice(rng, ["strong_retention", "gradual_drop"])
        comment_quality = _choice(rng, ["relevant_discussion", "short_but_relevant"])
    elif label == "suspicious":
        views = _bounded_lognormal(rng, 85000, 1.25, 2_500, 1_500_000)
        avg_multiplier = float(rng.uniform(4.0, 55.0))
        like_rate = float(rng.uniform(0.003, 0.035))
        comment_rate = float(rng.uniform(0.0, 0.0018))
        share_rate = float(rng.uniform(0.0, 0.0025))
        pattern = _choice(
            rng,
            ["step_like_batches", "flat_then_spike", "late_like_spike", "likes_freeze", "low_engagement_smooth"],
            [0.26, 0.24, 0.20, 0.17, 0.13],
        )
        retention = _choice(rng, ["sudden_drop", "low_watch_time", "flat_or_odd", "broken_or_missing"])
        comment_quality = _choice(rng, ["mixed", "generic_repeated", "emoji_spam", "unrelated"])
    else:
        views = _bounded_lognormal(rng, 220000, 1.05, 12_000, 3_500_000)
        avg_multiplier = float(rng.uniform(18.0, 200.0))
        like_rate = float(rng.uniform(0.0002, 0.010))
        comment_rate = float(rng.uniform(0.0, 0.00035))
        share_rate = float(rng.uniform(0.0, 0.0012))
        pattern = _choice(
            rng,
            ["vertical_spike_no_engagement", "flat_then_spike", "likes_freeze", "step_like_batches", "late_like_spike"],
            [0.34, 0.25, 0.18, 0.15, 0.08],
        )
        retention = _choice(rng, ["very_low_watch_time", "low_watch_time", "broken_or_missing", "unknown"])
        comment_quality = _choice(rng, ["none", "emoji_spam", "generic_repeated", "unrelated"])

    avg_views_30d = max(50, int(views / max(avg_multiplier, 0.1)))
    likes = int(round(views * like_rate))
    comments = int(round(views * comment_rate))
    shares = int(round(views * share_rate))
    if label == "botted" and rng.random() < 0.35:
        comments = 0
    if label != "clean" and rng.random() < 0.18:
        shares = min(shares, int(rng.integers(0, 4)))
    saves = int(round(max(0, views * rng.uniform(0.0002, 0.006))))
    reach_ratio = rng.uniform(0.55, 1.05) if label == "clean" else rng.uniform(0.12, 0.75)
    reach = int(round(views * reach_ratio))

    if label == "clean":
        completion_rate = float(rng.uniform(0.32, 0.82))
    elif label == "suspicious":
        completion_rate = float(rng.uniform(0.10, 0.48))
    else:
        completion_rate = float(rng.uniform(0.015, 0.28))
    avg_watch_time = round(video_length * completion_rate * rng.uniform(0.75, 1.1), 2)

    if label == "clean":
        tier1 = float(rng.uniform(0.35, 0.90))
        profile_traffic = float(rng.uniform(0.01, 0.18))
        search_traffic = float(rng.uniform(0.02, 0.25))
    else:
        tier1 = float(rng.uniform(0.02, 0.72))
        profile_traffic = float(rng.uniform(0.08, 0.90))
        search_traffic = float(rng.uniform(0.00, 0.60))
    fyp = float(max(0.0, min(0.98, 1 - profile_traffic - search_traffic + rng.uniform(-0.08, 0.08))))

    like_freeze_flag = pattern == "likes_freeze"
    late_like_spike_flag = pattern == "late_like_spike"
    no_matching_flag = label != "clean" and (like_rate < 0.015 or comment_rate < 0.001 or pattern in {"vertical_spike_no_engagement", "flat_then_spike"})
    max_jump_pct = {
        "smooth_gradual": rng.uniform(0.08, 0.35),
        "normal_viral_smooth": rng.uniform(0.10, 0.42),
        "low_engagement_smooth": rng.uniform(0.18, 0.65),
        "late_like_spike": rng.uniform(0.30, 1.20),
        "likes_freeze": rng.uniform(0.45, 1.50),
        "step_like_batches": rng.uniform(0.80, 2.20),
        "flat_then_spike": rng.uniform(1.20, 3.50),
        "vertical_spike_no_engagement": rng.uniform(1.80, 4.00),
    }[pattern]
    flatline_hours = int(rng.integers(0, 5)) if label == "clean" else int(rng.integers(4, 96))

    suspicious_comment_pct = 0.0 if label == "clean" else float(rng.uniform(0.10, 0.88))
    repeated_comment_pct = float(rng.uniform(0.0, 0.10)) if label == "clean" else float(rng.uniform(0.05, 0.65))
    bot_like_profile_pct = float(rng.uniform(0.0, 0.12)) if label == "clean" else float(rng.uniform(0.10, 0.75))
    score, risk_level, decision = _risk_metadata(label, rng)
    likely_fraud = _fraud_type(label, pattern, rng)

    reasons = []
    if max_jump_pct >= 1.0:
        reasons.append("large sudden view jump")
    if no_matching_flag:
        reasons.append("views rise without matching engagement")
    if comments <= 3 and views >= 10000:
        reasons.append("high views with almost no comments")
    if like_freeze_flag:
        reasons.append("likes freeze while views increase")
    if late_like_spike_flag:
        reasons.append("late like spike")
    if not reasons:
        reasons.append("metrics and graph pattern move proportionally")

    submission_id = f"GEN-{index:06d}"
    row = {
        "submission_id": submission_id,
        "platform": platform,
        "campaign_type": campaign,
        "days_live": days_live,
        "creator_followers": followers,
        "avg_views_30d": avg_views_30d,
        "video_length_sec": video_length,
        "views": views,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "saves": saves,
        "reach": reach,
        "avg_watch_time_sec": avg_watch_time,
        "completion_rate": round(completion_rate, 4),
        "tier1_audience_pct": round(tier1, 4),
        "profile_traffic_pct": round(profile_traffic, 4),
        "search_traffic_pct": round(search_traffic, 4),
        "fyp_reels_explore_pct": round(fyp, 4),
        "graph_pattern": pattern,
        "retention_pattern": retention,
        "max_view_jump_pct": round(max_jump_pct, 4),
        "flatline_hours_before_jump": flatline_hours,
        "like_freeze_flag": like_freeze_flag,
        "late_like_spike_flag": late_like_spike_flag,
        "no_matching_engagement_flag": no_matching_flag,
        "comment_quality": comment_quality,
        "suspicious_comment_pct": round(suspicious_comment_pct, 4),
        "repeated_comment_pct": round(repeated_comment_pct, 4),
        "bot_like_profile_pct": round(bot_like_profile_pct, 4),
        "views_to_avg_multiplier": round(safe_divide(views, avg_views_30d, 0), 4),
        "like_rate": round(safe_divide(likes, views, 0), 6),
        "comment_rate": round(safe_divide(comments, views, 0), 6),
        "share_rate": round(safe_divide(shares, views, 0), 6),
        "engagement_rate": round(safe_divide(likes + comments + shares, views, 0), 6),
        "reach_view_ratio": round(safe_divide(reach, views, 0), 6),
        "fraud_risk_score": score,
        "risk_level": risk_level,
        "label": label,
        "decision_recommendation": decision,
        "likely_fraud_type": likely_fraud,
        "reviewer_reason": "; ".join(reasons),
        "suggested_action": "Approve but monitor later." if label == "clean" else "Reject and ask for analytics in ticket." if label == "suspicious" else "Reject; escalate only if analytics confirms deliberate botting.",
        "source_type": "synthetic_generated",
        "source_note": "Generated supplemental training example. Not production validation evidence.",
    }

    graph_rows = []
    view_weights = _weights_for_pattern(rng, pattern, "views", label)
    like_weights = _weights_for_pattern(rng, pattern, "likes", label)
    comment_weights = _weights_for_pattern(rng, pattern, "comments", label)
    share_weights = _weights_for_pattern(rng, pattern, "shares", label)
    view_series = _cumulative(views, view_weights)
    like_series = _cumulative(likes, like_weights)
    comment_series = _cumulative(comments, comment_weights)
    share_series = _cumulative(shares, share_weights)
    for hour, view_value, like_value, comment_value, share_value in zip(
        HOURS,
        view_series,
        like_series,
        comment_series,
        share_series,
    ):
        graph_rows.append(
            {
                "submission_id": submission_id,
                "hour_since_post": int(hour),
                "views_cumulative": int(view_value),
                "likes_cumulative": int(like_value),
                "comments_cumulative": int(comment_value),
                "shares_cumulative": int(share_value),
                "graph_pattern": pattern,
                "label": label,
            }
        )
    return row, graph_rows


def generate_synthetic_data(
    n_rows: int = 10000,
    output_dir: Path = DATAS_DIR,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []

    labels = [LABELS[index % len(LABELS)] for index in range(n_rows)]
    rng.shuffle(labels)
    for idx, label in enumerate(labels, start=1):
        row, row_graph = _row_for_label(idx, label, rng)
        rows.append(row)
        graph_rows.extend(row_graph)

    submissions = pd.DataFrame(rows)
    graph = pd.DataFrame(graph_rows)

    original_path = output_dir / SUBMISSION_CSV
    if original_path.exists():
        original_columns = pd.read_csv(original_path, nrows=0).columns.tolist()
        submissions = submissions.reindex(columns=original_columns)

    graph_path = output_dir / GRAPH_CSV
    if graph_path.exists():
        graph_columns = pd.read_csv(graph_path, nrows=0).columns.tolist()
        graph = graph.reindex(columns=graph_columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    submissions.to_csv(output_dir / GENERATED_SUBMISSION_CSV, index=False)
    graph.to_csv(output_dir / GENERATED_GRAPH_CSV, index=False)
    return submissions, graph


def main() -> None:
    submissions, graph = generate_synthetic_data()
    print(f"Generated submissions: {len(submissions):,}")
    print(f"Generated graph rows: {len(graph):,}")
    print(f"Saved: {DATAS_DIR / GENERATED_SUBMISSION_CSV}")
    print(f"Saved: {DATAS_DIR / GENERATED_GRAPH_CSV}")


if __name__ == "__main__":
    main()

