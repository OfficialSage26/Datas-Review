"""Tikfly/RapidAPI integration for live TikTok review data."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from statistics import mean, median
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests


TIKFLY_BASE_URL = "https://tiktok-api23.p.rapidapi.com"
TIKFLY_HOST = "tiktok-api23.p.rapidapi.com"
MAX_HISTORY_COUNT = 35
VIDEO_ID_RE = re.compile(r"^\d{8,}$")
USER_POST_PATHS = {"/api/user/posts", "/api/user/oldest-posts"}


class TikflyError(RuntimeError):
    """Base exception for Tikfly integration failures."""


class TikflyInputError(TikflyError, ValueError):
    """Raised when reviewer input cannot be converted into a TikTok video ID."""


class TikflyAPIError(TikflyError):
    """Raised when RapidAPI/Tikfly returns an unusable response."""


def get_env_rapidapi_key() -> str:
    """Return the RapidAPI key from supported environment variable names."""
    return os.environ.get("RAPIDAPI_KEY", "").strip() or os.environ.get("X_RAPIDAPI_KEY", "").strip()


def clamp_history_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return MAX_HISTORY_COUNT
    return max(1, min(MAX_HISTORY_COUNT, count))


def extract_tiktok_video_id(value: str) -> str | None:
    """Extract a TikTok video ID from a direct ID or URL without network calls."""
    text = str(value or "").strip()
    if not text:
        return None
    if VIDEO_ID_RE.fullmatch(text):
        return text

    parsed = urlparse(text)
    if not parsed.netloc and not parsed.scheme:
        return None

    query = parse_qs(parsed.query)
    for key in ("videoId", "video_id", "item_id", "itemId", "aweme_id", "awemeId"):
        for candidate in query.get(key, []):
            if VIDEO_ID_RE.fullmatch(candidate):
                return candidate

    path_match = re.search(r"/video/(\d{8,})", parsed.path)
    if path_match:
        return path_match.group(1)

    for segment in reversed([part for part in parsed.path.split("/") if part]):
        if VIDEO_ID_RE.fullmatch(segment):
            return segment
    return None


def resolve_tiktok_video_id(
    value: str,
    session: requests.Session | None = None,
    timeout: float = 10.0,
) -> str:
    """Resolve direct TikTok URLs and redirected short links to a video ID."""
    direct = extract_tiktok_video_id(value)
    if direct:
        return direct

    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or "tiktok.com" not in parsed.netloc.lower():
        raise TikflyInputError("Enter a TikTok video URL or numeric video ID.")

    http = session or requests.Session()
    try:
        response = http.get(value, allow_redirects=True, timeout=timeout)
    except requests.RequestException as exc:
        raise TikflyInputError(f"Could not resolve TikTok link: {exc}") from exc

    final_url = str(getattr(response, "url", "") or value)
    resolved = extract_tiktok_video_id(final_url)
    if resolved:
        return resolved
    raise TikflyInputError("Could not find a TikTok video ID in that URL.")


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if not value:
            return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_timestamp(value: Any) -> datetime | None:
    timestamp = _coerce_int(value, default=0)
    if timestamp <= 0:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None


def _nested_get(row: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = row
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def extract_post_item(response: dict[str, Any]) -> dict[str, Any]:
    """Return the Tikfly item object from a post detail response."""
    candidates = [
        _nested_get(response, ("itemInfo", "itemStruct")),
        _nested_get(response, ("data", "itemInfo", "itemStruct")),
        _nested_get(response, ("data", "itemStruct")),
        response.get("itemStruct") if isinstance(response, dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    raise TikflyAPIError("Tikfly post detail response did not include itemInfo.itemStruct.")


def extract_item_list(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Return itemList from Tikfly user posts responses."""
    candidates = [
        _nested_get(response, ("data", "itemList")),
        response.get("itemList") if isinstance(response, dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
    return []


def _empty_item_list_response() -> dict[str, Any]:
    return {"data": {"cursor": "0", "hasMore": False, "itemList": []}}


def tikfly_item_to_submission(item: dict[str, Any], avg_views_30d: int = 0) -> dict[str, Any]:
    """Map one Tikfly item into the existing fraud model submission schema."""
    stats_v2 = item.get("statsV2") if isinstance(item.get("statsV2"), dict) else {}
    stats = item.get("stats") if isinstance(item.get("stats"), dict) else {}
    author = item.get("author") if isinstance(item.get("author"), dict) else {}
    author_stats_v2 = item.get("authorStatsV2") if isinstance(item.get("authorStatsV2"), dict) else {}
    author_stats = item.get("authorStats") if isinstance(item.get("authorStats"), dict) else {}
    video = item.get("video") if isinstance(item.get("video"), dict) else {}

    video_id = str(item.get("id") or video.get("id") or "")
    views = _coerce_int(stats_v2.get("playCount", stats.get("playCount")))
    likes = _coerce_int(stats_v2.get("diggCount", stats.get("diggCount")))
    comments = _coerce_int(stats_v2.get("commentCount", stats.get("commentCount")))
    shares = _coerce_int(stats_v2.get("shareCount", stats.get("shareCount")))
    saves = _coerce_int(stats_v2.get("collectCount", stats.get("collectCount")))

    return {
        "submission_id": f"TIKTOK-{video_id}" if video_id else "TIKTOK-LIVE",
        "platform": "TikTok",
        "source_type": "tikfly_api",
        "video_id": video_id,
        "tiktok_sec_uid": author.get("secUid") or "",
        "creator_username": author.get("uniqueId") or "",
        "creator_nickname": author.get("nickname") or "",
        "creator_followers": _coerce_int(author_stats_v2.get("followerCount", author_stats.get("followerCount"))),
        "views": views,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "saves": saves,
        "video_length_sec": _coerce_int(video.get("duration")),
        "avg_views_30d": int(avg_views_30d or 0),
        "graph_pattern": "unknown",
    }


def _unique_history_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item.get("id") or "")
        if item_id and item_id in seen:
            continue
        if item_id:
            seen.add(item_id)
        unique.append(item)
    return unique


def _item_views(item: dict[str, Any]) -> int:
    stats_v2 = item.get("statsV2") if isinstance(item.get("statsV2"), dict) else {}
    stats = item.get("stats") if isinstance(item.get("stats"), dict) else {}
    return _coerce_int(stats_v2.get("playCount", stats.get("playCount")))


def summarize_creator_history(
    recent_items: list[dict[str, Any]],
    oldest_items: list[dict[str, Any]],
    submitted_video_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build creator baseline metrics for the submitted video's prediction."""
    now = now or datetime.now(timezone.utc)
    submitted_id = str(submitted_video_id or "")
    recent_without_submission = [item for item in recent_items if str(item.get("id") or "") != submitted_id]
    all_history = _unique_history_items(recent_without_submission + oldest_items)
    all_history = [item for item in all_history if str(item.get("id") or "") != submitted_id]

    recent_30d = []
    for item in all_history:
        created_at = _coerce_timestamp(item.get("createTime"))
        if created_at and now - timedelta(days=30) <= created_at <= now + timedelta(minutes=5):
            recent_30d.append(item)

    if recent_30d:
        baseline_items = recent_30d
        baseline_source = "last_30_days"
    else:
        baseline_items = recent_without_submission
        baseline_source = "recent_posts" if recent_without_submission else "none"

    baseline_views = [_item_views(item) for item in baseline_items if _item_views(item) > 0]
    history_views = [_item_views(item) for item in all_history if _item_views(item) > 0]
    avg_views = int(round(mean(baseline_views))) if baseline_views else 0

    return {
        "avg_views_30d": avg_views,
        "baseline_source": baseline_source,
        "baseline_post_count": len(baseline_views),
        "recent_posts_count": len(recent_items),
        "oldest_posts_count": len(oldest_items),
        "history_posts_count": len(all_history),
        "max_history_views": max(history_views) if history_views else 0,
        "median_history_views": int(round(median(history_views))) if history_views else 0,
    }


def build_submission_from_tikfly(
    post_detail_response: dict[str, Any],
    recent_posts_response: dict[str, Any] | None = None,
    oldest_posts_response: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Convert Tikfly post detail and history responses into review-ready data."""
    post_item = extract_post_item(post_detail_response)
    recent_items = extract_item_list(recent_posts_response or {})
    oldest_items = extract_item_list(oldest_posts_response or {})
    video_id = str(post_item.get("id") or "")
    history_summary = summarize_creator_history(recent_items, oldest_items, video_id, now=now)
    submission = tikfly_item_to_submission(post_item, avg_views_30d=history_summary["avg_views_30d"])
    return {
        "video_id": submission["video_id"],
        "sec_uid": submission["tiktok_sec_uid"],
        "submission": submission,
        "history_summary": history_summary,
    }


class TikflyClient:
    """Small RapidAPI client for the Tikfly endpoints used by this app."""

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        base_url: str = TIKFLY_BASE_URL,
        timeout: float = 20.0,
    ) -> None:
        if not str(api_key or "").strip():
            raise TikflyAPIError("RAPIDAPI_KEY is not configured.")
        self.api_key = str(api_key).strip()
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        return {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": TIKFLY_HOST,
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.get(url, params=params, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TikflyAPIError(f"Tikfly request failed for {path}: {exc}") from exc

        if path in USER_POST_PATHS and getattr(response, "status_code", None) == 204:
            return _empty_item_list_response()

        try:
            data = response.json()
        except ValueError as exc:
            preview = str(getattr(response, "text", "") or "").strip().replace("\n", " ")[:160]
            detail = f"Tikfly returned non-JSON data for {path} (status {response.status_code})."
            if preview:
                detail = f"{detail} Response starts with: {preview}"
            raise TikflyAPIError(detail) from exc

        if not isinstance(data, dict):
            raise TikflyAPIError(f"Tikfly returned an unexpected response for {path}.")
        return data

    def get_post_detail(self, video_id: str) -> dict[str, Any]:
        return self._get("/api/post/detail", {"videoId": video_id})

    def get_user_posts(self, sec_uid: str, count: int = MAX_HISTORY_COUNT, cursor: str = "0") -> dict[str, Any]:
        return self._get("/api/user/posts", {"secUid": sec_uid, "count": clamp_history_count(count), "cursor": cursor})

    def get_user_oldest_posts(self, sec_uid: str, count: int = MAX_HISTORY_COUNT, cursor: str = "0") -> dict[str, Any]:
        return self._get("/api/user/oldest-posts", {"secUid": sec_uid, "count": clamp_history_count(count), "cursor": cursor})


def fetch_tiktok_review_context(
    tiktok_url_or_id: str,
    api_key: str,
    history_count: int = MAX_HISTORY_COUNT,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch Tikfly post data and creator history, then normalize it for prediction."""
    http = session or requests.Session()
    video_id = resolve_tiktok_video_id(tiktok_url_or_id, session=http)
    client = TikflyClient(api_key=api_key, session=http)
    post_detail = client.get_post_detail(video_id)
    post_item = extract_post_item(post_detail)
    author = post_item.get("author") if isinstance(post_item.get("author"), dict) else {}
    sec_uid = str(author.get("secUid") or "").strip()
    if not sec_uid:
        raise TikflyAPIError("Tikfly post detail response did not include author.secUid.")

    count = clamp_history_count(history_count)
    recent_posts = client.get_user_posts(sec_uid, count=count)
    history_warnings: list[str] = []
    try:
        oldest_posts = client.get_user_oldest_posts(sec_uid, count=count)
    except TikflyError as exc:
        oldest_posts = {"data": {"itemList": []}}
        history_warnings.append(f"Oldest posts unavailable: {exc}")

    context = build_submission_from_tikfly(post_detail, recent_posts, oldest_posts)
    if history_warnings:
        context["history_warnings"] = history_warnings
        context["history_summary"]["warnings"] = history_warnings
    context["requested_video_id"] = video_id
    context["history_count"] = count
    return context
