"""Strict Whop-style moderation decisions.

This module converts the model's internal classes and risk signals into the
two reviewer-facing actions used in the submission queue.
"""

from __future__ import annotations

from typing import Any


APPROVED = "Approved"
REJECT_FULL_ANALYTICS = "Reject: Send Full Analytics"

SUSPICIOUS_GRAPH_PATTERNS = {
    "flatline_then_vertical_spike",
    "vertical_spike_no_engagement",
    "flat_then_spike",
    "step_like_batches",
    "likes_freeze",
    "late_like_spike",
    "broken_or_inconsistent_graph",
    "possible_spike_or_flatline",
}

NORMAL_GRAPH_PATTERNS = {
    "smooth_gradual",
    "normal_viral_smooth",
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _prediction_value(prediction: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in prediction:
        return prediction[key]
    review = prediction.get("review") or {}
    return review.get(key, default)


def _metrics_from_submission(submission: dict[str, Any] | None) -> dict[str, float]:
    submission = submission or {}
    views = _num(submission.get("views"))
    likes = _num(submission.get("likes"))
    comments = _num(submission.get("comments"))
    shares = _num(submission.get("shares"))
    return {
        "views": views,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "like_rate": likes / views if views else 0.0,
        "comment_rate": comments / views if views else 0.0,
        "share_rate": shares / views if views else 0.0,
        "engagement_rate": (likes + comments + shares) / views if views else 0.0,
    }


def _missing_required_metrics(analysis: dict[str, Any] | None) -> list[str]:
    if not analysis:
        return []
    fields = analysis.get("fields") or {}
    required = ["views", "likes", "comments", "shares"]
    return [field for field in required if fields.get(field) is None]


def _confidence(risk_score: int, red_flags: list[str], missing: list[str], approved: bool) -> str:
    if approved and risk_score <= 18 and not red_flags and not missing:
        return "High"
    if not approved and (risk_score >= 75 or len(red_flags) >= 3 or missing):
        return "High"
    if risk_score >= 50 or len(red_flags) >= 2:
        return "Medium"
    return "Low"


def moderation_decision_from_prediction(
    prediction: dict[str, Any],
    screenshot_analysis: dict[str, Any] | None = None,
    submission: dict[str, Any] | None = None,
    campaign_requirements: str | None = None,
) -> dict[str, Any]:
    """Return the strict reviewer output for a prediction.

    The decision is intentionally conservative. Anything missing, unclear, or
    suspicious defaults to requesting full analytics.
    """
    risk_score = int(_num(prediction.get("risk_score"), 100))
    risk_level = str(prediction.get("risk_level", "Critical"))
    predicted_class = str(prediction.get("predicted_class", "Suspicious")).lower()
    review = prediction.get("review") or {}
    suspicious_signals = [
        signal
        for signal in review.get("suspicious_signals", [])
        if signal and signal != "No major fraud signal was triggered."
    ]
    missing_evidence = [
        item
        for item in review.get("missing_evidence_needed", [])
        if item and item != "No critical evidence missing from available fields."
    ]
    graph_pattern = str((submission or {}).get("graph_pattern") or (screenshot_analysis or {}).get("graph_shape") or "").lower()
    graph_vision = (screenshot_analysis or {}).get("graph_vision") or {}
    metrics = _metrics_from_submission(submission)
    missing_metrics = _missing_required_metrics(screenshot_analysis)

    red_flags: list[str] = []
    if missing_metrics:
        red_flags.append(f"Missing visible metric(s): {', '.join(missing_metrics)}.")
    if missing_evidence:
        red_flags.append("Full analytics proof is incomplete or not visible.")
    if campaign_requirements and len(campaign_requirements.strip()) > 0:
        red_flags.append("Campaign requirements cannot be fully confirmed from the screenshot alone.")
    if graph_pattern in SUSPICIOUS_GRAPH_PATTERNS:
        red_flags.append("Graph shape is suspicious or not organic.")
    if graph_vision.get("confidence", 0) and _num(graph_vision.get("confidence")) < 0.45:
        red_flags.append("Graph detection confidence is low.")
    if metrics["views"] >= 10000 and metrics["comments"] == 0:
        red_flags.append("High views with zero comments.")
    elif metrics["views"] >= 10000 and metrics["comments"] <= 3:
        red_flags.append("High views with very few comments.")
    if metrics["views"] >= 100000 and metrics["shares"] <= 3:
        red_flags.append("High views with very low shares.")
    if metrics["views"] >= 10000 and metrics["engagement_rate"] < 0.01:
        red_flags.append("Visible engagement is too low for the view count.")
    if risk_level in {"Medium", "High", "Critical"} or risk_score >= 25:
        red_flags.append(f"Model risk is {risk_level.lower()} at {risk_score}/100.")
    if predicted_class in {"suspicious", "botted"}:
        red_flags.append(f"Model class is {predicted_class}.")
    if (screenshot_analysis or {}).get("error"):
        red_flags.append("Screenshot could not be fully analyzed.")

    deduped_red_flags = list(dict.fromkeys(red_flags))
    graph_is_normal = graph_pattern in NORMAL_GRAPH_PATTERNS or graph_pattern in {"", "unknown"}
    metrics_are_proportional = (
        metrics["views"] > 0
        and metrics["engagement_rate"] >= 0.02
        and (metrics["views"] < 10000 or metrics["comments"] >= 4)
    )
    approved = (
        risk_score <= 24
        and predicted_class == "clean"
        and metrics_are_proportional
        and graph_is_normal
        and not deduped_red_flags
        and not missing_metrics
        and not missing_evidence
    )

    decision = APPROVED if approved else REJECT_FULL_ANALYTICS
    if approved:
        reason = [
            "Metrics are proportionate to the view count.",
            "Graph behavior appears organic.",
            "No major red flags are visible.",
        ]
        mod_note = "Approved based on visible proportional engagement and organic analytics."
        red_flags_output = ["No major red flags found."]
    else:
        reason = deduped_red_flags[:4] or ["Analytics are not sufficient to approve from the screenshot alone."]
        mod_note = "Please send full analytics before this submission can be approved."
        red_flags_output = deduped_red_flags or ["Analytics are incomplete or unclear."]

    return {
        "decision": decision,
        "confidence": _confidence(risk_score, deduped_red_flags, missing_metrics + missing_evidence, approved),
        "reason": reason[:4],
        "red_flags": red_flags_output,
        "mod_note": mod_note,
        "internal": {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "predicted_class": prediction.get("predicted_class"),
            "model_decision": prediction.get("decision"),
        },
    }


def format_moderation_output(result: dict[str, Any]) -> str:
    lines = [
        f"Decision: {result['decision']}",
        "",
        f"Confidence: {result['confidence']}",
        "",
        "Reason:",
    ]
    lines.extend(f"- {item}" for item in result.get("reason", []))
    lines.extend(["", "Red Flags:"])
    lines.extend(f"- {item}" for item in result.get("red_flags", []))
    lines.extend(["", "Mod Note:", f"- {result['mod_note']}"])
    return "\n".join(lines)

