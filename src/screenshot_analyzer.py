"""Screenshot and screen-vision analyzer for dashboard submissions.

The analyzer uses optional OCR and OpenCV. If either dependency or the local
Tesseract executable is unavailable, it returns a safe partial result instead
of crashing. Extracted metrics can be sent directly into the fraud prediction
model with ``analyze_dashboard_screenshot``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


METRIC_ALIASES = {
    "views": ["views", "view count", "plays", "impressions"],
    "likes": ["likes", "hearts"],
    "comments": ["comments", "replies"],
    "shares": ["shares", "reposts"],
    "saves": ["saves", "bookmarks"],
    "payout": ["payout", "reward", "earnings"],
    "status": ["status", "decision"],
}

PLATFORM_KEYWORDS = {
    "TikTok": ["tiktok", "tik tok"],
    "Instagram": ["instagram", "reels", "reel"],
    "YouTube Shorts": ["youtube", "shorts", "yt shorts"],
}


def _number(value: str) -> int | float | str:
    cleaned = value.replace(",", "").replace("$", "").strip().lower()
    multiplier = 1
    if cleaned.endswith("k"):
        multiplier = 1_000
        cleaned = cleaned[:-1]
    elif cleaned.endswith("m"):
        multiplier = 1_000_000
        cleaned = cleaned[:-1]
    elif cleaned.endswith("b"):
        multiplier = 1_000_000_000
        cleaned = cleaned[:-1]
    cleaned = cleaned.strip()
    try:
        number = float(cleaned) * multiplier
    except ValueError:
        return value.strip()
    return int(number) if number.is_integer() else round(number, 2)


def _metric_regex(alias: str) -> list[str]:
    escaped = re.escape(alias)
    number = r"(\$?\d+(?:[,.]\d+)*(?:\.\d+)?\s*[kKmMbB]?)"
    return [
        rf"{escaped}\s*[:#\-]?\s*{number}",
        rf"{number}\s+{escaped}",
    ]


def _extract_metric(text: str, aliases: list[str]) -> int | float | str | None:
    normalized = re.sub(r"\s+", " ", text)
    for alias in aliases:
        for pattern in _metric_regex(alias):
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                value = match.group(1)
                if re.search(r"\d", value):
                    return _number(value)
    return None


def _extract_metric_from_lines(lines: list[str], aliases: list[str]) -> int | float | str | None:
    for index, line in enumerate(lines):
        lowered = line.lower()
        if not any(alias in lowered for alias in aliases):
            continue
        tail = re.sub("|".join(re.escape(alias) for alias in aliases), "", line, flags=re.IGNORECASE).strip(" :#-\t")
        if re.search(r"\d", tail):
            match = re.search(r"\$?\d+(?:[,.]\d+)*(?:\.\d+)?\s*[kKmMbB]?", tail)
            if match:
                return _number(match.group(0))
        for next_line in lines[index + 1 : index + 4]:
            match = re.search(r"\$?\d+(?:[,.]\d+)*(?:\.\d+)?\s*[kKmMbB]?", next_line)
            if match:
                return _number(match.group(0))
    return None


def _extract_status(text: str) -> str | None:
    match = re.search(r"(?:status|decision)\s*[:#\-]?\s*([A-Za-z ]{3,40})", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if re.search(r"\bapprove\b|\bapproved\b", text, flags=re.IGNORECASE):
        return "Approve visible"
    if re.search(r"\breject\b|\brejected\b", text, flags=re.IGNORECASE):
        return "Reject visible"
    return None


def _extract_platform(text: str) -> str | None:
    lowered = text.lower()
    for platform, keywords in PLATFORM_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return platform
    return None


def extract_visible_metrics(text: str) -> dict[str, Any]:
    """Extract visible metric fields from OCR text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    fields: dict[str, Any] = {}
    for field, aliases in METRIC_ALIASES.items():
        if field == "status":
            fields[field] = _extract_status(text)
        else:
            fields[field] = _extract_metric(text, aliases)
            if fields[field] is None:
                fields[field] = _extract_metric_from_lines(lines, aliases)
    fields["platform"] = _extract_platform(text)
    return fields


def _ocr_image(image_path: Path, timeout_seconds: int = 6, max_dimension: int = 1600) -> dict[str, Any]:
    result = {
        "ocr_available": False,
        "text": "",
        "error": None,
    }
    try:
        from PIL import Image, ImageOps
        import pytesseract

        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image.thumbnail((max_dimension, max_dimension))
        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image)
        result["ocr_available"] = True
        result["text"] = pytesseract.image_to_string(image, timeout=timeout_seconds)
    except Exception as exc:
        result["error"] = f"OCR failed or is not configured: {exc}"
    return result


def _smooth(values: list[float], window: int = 3) -> list[float]:
    if len(values) < window:
        return values
    smoothed: list[float] = []
    half = window // 2
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def _classify_graph_points(points: list[tuple[float, float]]) -> dict[str, Any]:
    if len(points) < 8:
        return {
            "graph_pattern": "unknown",
            "confidence": 0.0,
            "reason": "Not enough graph points were detected.",
        }

    points = sorted(points, key=lambda item: item[0])
    y_values = _smooth([point[1] for point in points], window=5)
    min_y = min(y_values)
    max_y = max(y_values)
    span = max(max_y - min_y, 1e-6)
    normalized = [(value - min_y) / span for value in y_values]
    deltas = [normalized[index + 1] - normalized[index] for index in range(len(normalized) - 1)]
    positive_deltas = [max(delta, 0.0) for delta in deltas]
    max_delta = max(positive_deltas) if positive_deltas else 0.0
    max_delta_index = positive_deltas.index(max_delta) if positive_deltas else 0
    total_positive = sum(positive_deltas) or 1e-6
    top_two_share = sum(sorted(positive_deltas, reverse=True)[:2]) / total_positive
    first_half_range = max(normalized[: len(normalized) // 2]) - min(normalized[: len(normalized) // 2])
    pre_spike_range = max(normalized[: max(max_delta_index, 1)]) - min(normalized[: max(max_delta_index, 1)])
    jump_count = sum(1 for delta in positive_deltas if delta >= max(0.08, max_delta * 0.45))
    negative_count = sum(1 for delta in deltas if delta < -0.08)
    volatility = sum(abs(delta) for delta in deltas) / max(len(deltas), 1)

    if max_delta >= 0.38 and (first_half_range <= 0.15 or pre_spike_range <= 0.12):
        pattern = "flatline_then_vertical_spike"
        confidence = min(0.95, 0.55 + max_delta + (0.20 if top_two_share >= 0.70 else 0))
        reason = "Detected a low-movement region followed by one dominant upward jump."
    elif max_delta >= 0.42:
        pattern = "vertical_spike_no_engagement"
        confidence = min(0.90, 0.50 + max_delta)
        reason = "Detected one dominant vertical movement in the graph."
    elif jump_count >= 3 and top_two_share < 0.75:
        pattern = "step_like_batches"
        confidence = min(0.85, 0.45 + jump_count * 0.08)
        reason = "Detected multiple separated upward jumps with plateaus."
    elif negative_count >= 3 or volatility >= 0.12:
        pattern = "broken_or_inconsistent_graph"
        confidence = min(0.75, 0.45 + volatility)
        reason = "Detected irregular graph movement."
    elif max_delta <= 0.22:
        pattern = "smooth_gradual"
        confidence = 0.70
        reason = "Detected gradual graph movement without a dominant spike."
    else:
        pattern = "normal_viral_smooth"
        confidence = 0.62
        reason = "Detected mostly smooth upward movement."

    return {
        "graph_pattern": pattern,
        "confidence": round(float(confidence), 3),
        "reason": reason,
        "points_detected": len(points),
        "max_normalized_jump": round(float(max_delta), 4),
        "top_two_jump_share": round(float(top_two_share), 4),
        "jump_count": int(jump_count),
    }


def _detect_graph_with_cv(image_path: Path) -> dict[str, Any]:
    result = {
        "cv_available": False,
        "graph_pattern": "unknown",
        "confidence": 0.0,
        "reason": "Computer vision was not run.",
        "points_detected": 0,
        "error": None,
    }
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        result["error"] = f"OpenCV is not installed or could not import: {exc}"
        return result

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            result["error"] = "OpenCV could not read the image file."
            return result
        result["cv_available"] = True
        height, width = image.shape[:2]
        max_dimension = 1800
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
            height, width = image.shape[:2]

        # Most dashboard graphs sit below metric cards, so scan the lower/middle region.
        y0 = int(height * 0.22)
        y1 = int(height * 0.95)
        x0 = int(width * 0.05)
        x1 = int(width * 0.98)
        roi = image[y0:y1, x0:x1]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        saturation_mask = cv2.inRange(hsv[:, :, 1], 45, 255)
        dark_mask = cv2.inRange(gray, 0, 185)
        edge_mask = cv2.Canny(gray, 60, 160)
        mask = cv2.bitwise_or(cv2.bitwise_and(saturation_mask, dark_mask), edge_mask)
        kernel = np.ones((2, 2), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            result["reason"] = "No graph-like contours were found."
            return result

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if area < 20 or w < width * 0.08 or h < 8:
                continue
            candidates.append((area * w, contour, (x, y, w, h)))
        if not candidates:
            result["reason"] = "No graph-like contour was large enough to classify."
            return result

        _, contour, bbox = max(candidates, key=lambda item: item[0])
        x, y, w, h = bbox
        contour_points = contour.reshape(-1, 2)
        bins = min(80, max(16, w // 8))
        points: list[tuple[float, float]] = []
        for bin_index in range(bins):
            left = x + (w * bin_index / bins)
            right = x + (w * (bin_index + 1) / bins)
            subset = contour_points[(contour_points[:, 0] >= left) & (contour_points[:, 0] < right)]
            if len(subset) == 0:
                continue
            # Image y increases downward; invert it so larger means more views.
            x_mid = (left + right) / 2
            y_value = 1.0 - float(np.median(subset[:, 1]) / max(roi.shape[0], 1))
            points.append((x_mid, y_value))

        classified = _classify_graph_points(points)
        classified["action_visible"] = _detect_visible_action_label(image)
        result.update(classified)
        result["points_detected"] = classified.get("points_detected", len(points))
    except Exception as exc:
        result["error"] = f"Graph computer vision failed: {exc}"
    return result


def _detect_visible_action_label(image: Any) -> str | None:
    try:
        import cv2
        import numpy as np

        height, width = image.shape[:2]
        bottom = image[int(height * 0.82) : height, 0:width]
        hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([95, 255, 255]))
        red_mask_1 = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([12, 255, 255]))
        red_mask_2 = cv2.inRange(hsv, np.array([165, 35, 35]), np.array([179, 255, 255]))
        green_ratio = float(np.count_nonzero(green_mask)) / max(green_mask.size, 1)
        red_ratio = float(np.count_nonzero(red_mask_1 | red_mask_2)) / max(red_mask_1.size, 1)
        if green_ratio > 0.02 and green_ratio > red_ratio:
            return "Approve button visible"
        if red_ratio > 0.02 and red_ratio > green_ratio:
            return "Reject button visible"
    except Exception:
        return None
    return None


def analyze_screenshot(path: str | Path) -> dict[str, Any]:
    """Extract OCR metrics and graph shape from a dashboard screenshot."""
    image_path = Path(path)
    result: dict[str, Any] = {
        "screenshot_path": str(image_path),
        "ocr_available": False,
        "cv_available": False,
        "text": "",
        "fields": {
            "views": None,
            "likes": None,
            "comments": None,
            "shares": None,
            "saves": None,
            "payout": None,
            "status": None,
            "platform": None,
        },
        "graph_shape": "unknown",
        "graph_vision": {},
        "missing_fields": ["views", "likes", "comments", "shares", "saves", "payout", "status", "platform"],
        "error": None,
    }
    if not image_path.exists():
        result["error"] = "Screenshot path does not exist."
        return result

    ocr = _ocr_image(image_path)
    result["ocr_available"] = bool(ocr["ocr_available"])
    result["text"] = ocr["text"]
    if ocr["error"]:
        result["error"] = ocr["error"]
    if result["text"]:
        result["fields"] = extract_visible_metrics(result["text"])

    graph = _detect_graph_with_cv(image_path)
    result["cv_available"] = bool(graph["cv_available"])
    result["graph_shape"] = graph.get("graph_pattern", "unknown")
    result["graph_vision"] = graph
    if graph.get("error") and not result["error"]:
        result["error"] = graph["error"]

    result["missing_fields"] = [key for key, value in result["fields"].items() if value is None]
    return result


def extracted_metrics_to_submission(
    analysis: dict[str, Any],
    platform: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert screenshot analysis into a prediction-ready submission dict."""
    fields = dict(analysis.get("fields") or {})
    submission = {
        "platform": platform or fields.get("platform") or "unknown",
        "views": fields.get("views") or 0,
        "likes": fields.get("likes") or 0,
        "comments": fields.get("comments") or 0,
        "shares": fields.get("shares") or 0,
        "saves": fields.get("saves") or 0,
        "graph_pattern": analysis.get("graph_shape") or "unknown",
        "source_type": "dashboard_screenshot",
    }
    if extra_fields:
        submission.update(extra_fields)
    return submission


def analyze_dashboard_screenshot(
    path: str | Path,
    platform: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze a screenshot and run the extracted values through the fraud model."""
    from .moderation import moderation_decision_from_prediction
    from .predict import predict_one

    analysis = analyze_screenshot(path)
    submission = extracted_metrics_to_submission(analysis, platform=platform, extra_fields=extra_fields)
    prediction = predict_one(submission)
    moderation = moderation_decision_from_prediction(
        prediction,
        screenshot_analysis=analysis,
        submission=submission,
        campaign_requirements=(extra_fields or {}).get("campaign_requirements") if extra_fields else None,
    )
    return {
        "screenshot_analysis": analysis,
        "submission": submission,
        "prediction": prediction,
        "moderation": moderation,
    }


if __name__ == "__main__":
    import json
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else ""
    platform_arg = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(analyze_dashboard_screenshot(target, platform=platform_arg), indent=2))
