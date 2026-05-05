"""Human-readable reviewer explanations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .utils import clean_text


def _get(row: dict[str, Any] | pd.Series, key: str, default: Any = None) -> Any:
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)


def _rate(row: dict[str, Any] | pd.Series, key: str) -> float:
    try:
        return float(_get(row, key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def calculated_ratios(row: dict[str, Any] | pd.Series) -> dict[str, float]:
    return {
        "like_rate": round(_rate(row, "like_rate"), 6),
        "comment_rate": round(_rate(row, "comment_rate"), 6),
        "share_rate": round(_rate(row, "share_rate"), 6),
        "total_visible_engagement_rate": round(_rate(row, "total_engagement_rate") or _rate(row, "engagement_rate"), 6),
    }


def suspicious_signals(row: dict[str, Any] | pd.Series) -> list[str]:
    signals: list[str] = []
    views = _rate(row, "views")
    comments = _rate(row, "comments")
    graph_pattern = clean_text(_get(row, "graph_pattern", ""), "").lower()

    if _rate(row, "high_views_low_likes"):
        signals.append("High views with unusually low like rate.")
    if _rate(row, "high_views_zero_comments"):
        signals.append("High views with zero comments.")
    elif _rate(row, "high_views_low_comments") or (views >= 10000 and comments <= 3):
        signals.append("High views with very few comments.")
    if _rate(row, "high_views_low_shares"):
        signals.append("High views with very low shares.")
    if graph_pattern in {"vertical_spike_no_engagement", "flat_then_spike", "flatline_then_vertical_spike"}:
        signals.append("Graph suggests a sudden view spike after low activity.")
    if _rate(row, "vertical_spike_flag") or _rate(row, "view_jump_ratio") >= 0.45:
        signals.append("A large share of total views arrived in one interval.")
    if _rate(row, "pre_spike_flatline_hours") >= 4 or _rate(row, "flatline_then_spike_score") >= 55:
        signals.append("Graph shows flat or near-flat growth before a later spike.")
    if graph_pattern == "step_like_batches" or _rate(row, "step_like_growth"):
        signals.append("Graph growth appears step-like or batched.")
    if _rate(row, "likes_freeze_during_view_growth") or bool(_get(row, "like_freeze_flag", False)):
        signals.append("Likes appear frozen while views increase.")
    if _rate(row, "late_like_spike") or bool(_get(row, "late_like_spike_flag", False)):
        signals.append("Likes appear to spike late instead of rising naturally with views.")
    if _rate(row, "views_engagement_divergence") >= 0.2 or bool(_get(row, "no_matching_engagement_flag", False)):
        signals.append("Views rise without matching visible engagement.")
    if _rate(row, "spike_engagement_gap_ratio") >= 0.6 or _rate(row, "engagement_lag_score") >= 60:
        signals.append("The largest view movement has weak or delayed engagement response.")
    if _rate(row, "late_like_spike_strength") >= 0.25 and _rate(row, "view_like_peak_lag_hours") >= 4:
        signals.append("A large like spike appears after the main view spike.")
    if _rate(row, "suspicious_comment_pct") >= 0.4:
        signals.append("Comment quality signals are suspicious.")
    if _rate(row, "views_to_avg_multiplier") >= 25:
        signals.append("Views are far above the creator's recent average.")
    return signals or ["No major fraud signal was triggered."]


def normal_signals(row: dict[str, Any] | pd.Series) -> list[str]:
    signals: list[str] = []
    graph_pattern = clean_text(_get(row, "graph_pattern", ""), "").lower()
    if graph_pattern in {"smooth_gradual", "normal_viral_smooth"}:
        signals.append("Graph pattern looks gradual or naturally viral.")
    if _rate(row, "total_engagement_rate") >= 0.02:
        signals.append("Visible engagement is proportional to views.")
    if _rate(row, "comment_growth_alignment") >= 55 or _rate(row, "share_growth_alignment") >= 55:
        signals.append("Engagement movement aligns with view growth.")
    if _rate(row, "completion_rate") >= 0.30:
        signals.append("Retention is not unusually low.")
    return signals or ["No strong normalizing signal was available."]


def graph_analysis(row: dict[str, Any] | pd.Series) -> str:
    pattern = clean_text(_get(row, "graph_pattern", "unknown"))
    graph_score = _rate(row, "graph_suspicion_score")
    max_jump = _rate(row, "max_view_jump")
    jump_ratio = _rate(row, "view_jump_ratio")
    spikes = _rate(row, "number_of_spikes")
    flatline = _rate(row, "flatline_duration") or _rate(row, "flatline_hours_before_jump")
    pre_spike_flatline = _rate(row, "pre_spike_flatline_hours")
    robust_spike_ratio = _rate(row, "robust_spike_ratio")
    top_two_share = _rate(row, "top_two_view_jump_share")
    divergence = _rate(row, "views_engagement_divergence")
    spike_gap = _rate(row, "spike_engagement_gap_ratio")
    like_lag = _rate(row, "view_like_peak_lag_hours")
    late_like_strength = _rate(row, "late_like_spike_strength")

    if pattern == "unknown" and graph_score == 0:
        return "No graph time-series was provided, so graph risk is based only on any submitted graph pattern field."
    if max_jump == 0 and jump_ratio == 0 and graph_score > 0:
        return (
            f"Graph pattern is {pattern}. Graph suspicion score is {graph_score:.0f}/100. "
            "No graph time-series records were supplied with this prediction, so spike size and timing could not be recalculated."
        )
    return (
        f"Graph pattern is {pattern}. Graph suspicion score is {graph_score:.0f}/100. "
        f"Max view jump is {max_jump:.0f} views ({jump_ratio:.1%} of final views), "
        f"with {spikes:.0f} spike interval(s), about {flatline:.0f} flatline hour(s), "
        f"and {pre_spike_flatline:.0f} flatline hour(s) immediately before the largest spike. "
        f"Robust spike ratio is {robust_spike_ratio:.1f}x, top-two jumps contain {top_two_share:.1%} of views, "
        f"view-engagement divergence is {divergence:.1%}, spike engagement gap is {spike_gap:.1%}, "
        f"and the largest like spike lag is {like_lag:.0f} hour(s) with {late_like_strength:.1%} of likes."
    )


def missing_evidence(row: dict[str, Any] | pd.Series) -> list[str]:
    missing: list[str] = []
    if _get(row, "traffic_source", None) in {None, "", "unknown"} and not any(
        key in row for key in ["profile_traffic_pct", "search_traffic_pct", "fyp_reels_explore_pct"]
    ):
        missing.append("Traffic source breakdown")
    if not (_get(row, "avg_watch_time_sec", None) or _get(row, "watch_time", None) or _get(row, "completion_rate", None)):
        missing.append("Watch time or retention")
    if _get(row, "audience_location", None) in {None, "", "unknown"} and "tier1_audience_pct" not in row:
        missing.append("Audience location or geography")
    if _get(row, "creator_followers", None) in {None, "", "unknown"}:
        missing.append("Account history or follower baseline")
    if _get(row, "graph_pattern", None) in {None, "", "unknown"} and not _get(row, "max_view_jump", None):
        missing.append("View graph screenshot or time-series")
    return missing or ["No critical evidence missing from available fields."]


def creator_facing_reason(decision: str, signals: list[str]) -> str:
    if decision == "Approve":
        return "Submission currently appears consistent with normal engagement patterns."
    if decision == "Needs Manual Review":
        return "Engagement pattern needs manual review before approval."
    if decision == "Reject and Ask for Analytics in Ticket":
        return "Engagement pattern requires additional analytics review."
    if decision == "Escalate":
        return "Engagement pattern requires senior review due to repeated or severe inconsistencies."
    return "Engagement pattern is not consistent enough for approval."


def reviewer_reasoning(
    row: dict[str, Any] | pd.Series,
    decision: str,
    risk_score: int,
    predicted_class: str,
    signals: list[str],
) -> str:
    signal_text = " ".join(signals[:3])
    return (
        f"The system classifies this submission as {predicted_class} with a {risk_score}/100 risk score. "
        f"The decision is based on combined evidence across ratios, graph behavior, and available analytics. {signal_text}"
    ).strip()


def final_verdict(decision: str, risk_level: str, predicted_class: str) -> str:
    return (
        f"Probability-based conclusion: this looks {risk_level.lower()} risk and is most consistent "
        f"with a {predicted_class.lower()} submission. Recommended action: {decision}."
    )


def generate_review(
    row: dict[str, Any] | pd.Series,
    decision: str,
    risk_score: int,
    risk_level: str,
    predicted_class: str,
    fraud_types: dict[str, str],
) -> dict[str, Any]:
    suspicious = suspicious_signals(row)
    normal = normal_signals(row)
    missing = missing_evidence(row)
    creator_reason = creator_facing_reason(decision, suspicious)
    reasoning = reviewer_reasoning(row, decision, risk_score, predicted_class, suspicious)
    return {
        "title": "AI Video Fraud Review",
        "decision_recommendation": decision,
        "fraud_risk_score": risk_score,
        "risk_level": risk_level,
        "predicted_class": predicted_class,
        "calculated_ratios": calculated_ratios(row),
        "graph_analysis": graph_analysis(row),
        "suspicious_signals": suspicious,
        "normal_signals": normal,
        "likely_fraud_types": fraud_types,
        "reviewer_reasoning": reasoning,
        "missing_evidence_needed": missing,
        "creator_facing_rejection_reason": creator_reason,
        "final_verdict": final_verdict(decision, risk_level, predicted_class),
    }


def format_review_text(review: dict[str, Any]) -> str:
    ratios = review["calculated_ratios"]
    fraud_types = review["likely_fraud_types"]
    lines = [
        "---",
        "AI Video Fraud Review",
        "",
        "Decision Recommendation:",
        str(review["decision_recommendation"]),
        "",
        "Fraud Risk Score:",
        str(review["fraud_risk_score"]),
        "",
        "Risk Level:",
        str(review["risk_level"]),
        "",
        "Predicted Class:",
        str(review["predicted_class"]),
        "",
        "Calculated Ratios:",
        f"- Like rate: {ratios['like_rate']}",
        f"- Comment rate: {ratios['comment_rate']}",
        f"- Share rate: {ratios['share_rate']}",
        f"- Total visible engagement rate: {ratios['total_visible_engagement_rate']}",
        "",
        "Graph Analysis:",
        str(review["graph_analysis"]),
        "",
        "Suspicious Signals:",
    ]
    lines.extend(f"- {item}" for item in review["suspicious_signals"])
    lines.append("")
    lines.append("Normal Signals:")
    lines.extend(f"- {item}" for item in review["normal_signals"])
    lines.append("")
    lines.append("Likely Fraud Types:")
    lines.extend(f"- {key}: {value}" for key, value in fraud_types.items())
    lines.extend(
        [
            "",
            "Reviewer Reasoning:",
            str(review["reviewer_reasoning"]),
            "",
            "Missing Evidence Needed:",
        ]
    )
    lines.extend(f"- {item}" for item in review["missing_evidence_needed"])
    lines.extend(
        [
            "",
            "Creator-Facing Rejection Reason:",
            str(review["creator_facing_rejection_reason"]),
            "",
            "Final Verdict:",
            str(review["final_verdict"]),
            "---",
        ]
    )
    return "\n".join(lines)
