from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.tikfly_client import (
    build_submission_from_tikfly,
    extract_tiktok_video_id,
    fetch_tiktok_review_context,
    resolve_tiktok_video_id,
    summarize_creator_history,
    TikflyInputError,
)


def _item(video_id: str, play_count: int, create_time: int | None = None) -> dict:
    return {
        "id": video_id,
        "createTime": create_time or 0,
        "author": {
            "secUid": "MS4wLjABAAAAexample",
            "uniqueId": "creator",
            "nickname": "Creator",
        },
        "authorStatsV2": {
            "followerCount": "12345",
        },
        "statsV2": {
            "playCount": str(play_count),
            "diggCount": "456",
            "commentCount": "12",
            "shareCount": "34",
            "collectCount": "56",
        },
        "video": {
            "duration": 42,
        },
    }


def test_extract_tiktok_video_id_from_direct_inputs() -> None:
    video_id = "7306132438047116586"
    assert extract_tiktok_video_id(video_id) == video_id
    assert extract_tiktok_video_id(f"https://www.tiktok.com/@creator/video/{video_id}") == video_id
    assert extract_tiktok_video_id(f"https://www.tiktok.com/share/video/{video_id}?region=US") == video_id
    assert extract_tiktok_video_id(f"https://www.tiktok.com/?item_id={video_id}") == video_id
    assert extract_tiktok_video_id("not a tiktok url") is None


def test_resolve_tiktok_video_id_uses_redirected_short_link() -> None:
    video_id = "7306132438047116586"

    class FakeSession:
        def get(self, url: str, **kwargs):
            return SimpleNamespace(url=f"https://www.tiktok.com/@creator/video/{video_id}")

    assert resolve_tiktok_video_id("https://vm.tiktok.com/ZMexample/", session=FakeSession()) == video_id


def test_resolve_tiktok_video_id_rejects_invalid_input() -> None:
    with pytest.raises(TikflyInputError):
        resolve_tiktok_video_id("https://example.com/not-tiktok")


def test_build_submission_from_tikfly_maps_post_stats_and_recent_baseline() -> None:
    now = datetime(2026, 5, 7, tzinfo=timezone.utc)
    submitted_id = "7306132438047116586"
    post_response = {"itemInfo": {"itemStruct": _item(submitted_id, 50_000)}}
    recent_response = {
        "data": {
            "itemList": [
                _item(submitted_id, 50_000, int((now - timedelta(days=1)).timestamp())),
                _item("7306132438047116001", 10_000, int((now - timedelta(days=5)).timestamp())),
                _item("7306132438047116002", 20_000, int((now - timedelta(days=20)).timestamp())),
            ]
        }
    }
    oldest_response = {
        "data": {
            "itemList": [
                _item("7306132438047115001", 1_000, int((now - timedelta(days=180)).timestamp())),
            ]
        }
    }

    context = build_submission_from_tikfly(post_response, recent_response, oldest_response, now=now)
    submission = context["submission"]
    history = context["history_summary"]

    assert submission["platform"] == "TikTok"
    assert submission["source_type"] == "tikfly_api"
    assert submission["views"] == 50_000
    assert submission["likes"] == 456
    assert submission["comments"] == 12
    assert submission["shares"] == 34
    assert submission["saves"] == 56
    assert submission["video_length_sec"] == 42
    assert submission["creator_followers"] == 12345
    assert submission["graph_pattern"] == "unknown"
    assert submission["avg_views_30d"] == 15_000
    assert history["baseline_source"] == "last_30_days"
    assert history["baseline_post_count"] == 2


def test_history_baseline_falls_back_to_recent_posts_when_no_30_day_posts() -> None:
    now = datetime(2026, 5, 7, tzinfo=timezone.utc)
    submitted_id = "7306132438047116586"
    recent_items = [
        _item(submitted_id, 50_000, int((now - timedelta(days=90)).timestamp())),
        _item("7306132438047116001", 3_000, int((now - timedelta(days=90)).timestamp())),
        _item("7306132438047116002", 9_000, int((now - timedelta(days=120)).timestamp())),
    ]
    oldest_items = [
        _item("7306132438047115001", 50, int((now - timedelta(days=600)).timestamp())),
    ]

    history = summarize_creator_history(recent_items, oldest_items, submitted_id, now=now)

    assert history["avg_views_30d"] == 6_000
    assert history["baseline_source"] == "recent_posts"
    assert history["baseline_post_count"] == 2


def test_fetch_context_treats_oldest_posts_204_as_empty_history() -> None:
    submitted_id = "7306132438047116586"
    now = datetime.now(timezone.utc)

    class FakeResponse:
        def __init__(self, data: dict | None = None, text: str = "", status_code: int = 200) -> None:
            self.data = data or {}
            self.text = text
            self.status_code = status_code

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.data

    class FakeSession:
        def get(self, url: str, **kwargs):
            if url.endswith("/api/post/detail"):
                return FakeResponse({"itemInfo": {"itemStruct": _item(submitted_id, 50_000)}})
            if url.endswith("/api/user/posts"):
                return FakeResponse(
                    {
                        "data": {
                            "itemList": [
                                _item(submitted_id, 50_000, int((now - timedelta(days=1)).timestamp())),
                                _item("7306132438047116001", 10_000, int((now - timedelta(days=2)).timestamp())),
                            ]
                        }
                    }
                )
            if url.endswith("/api/user/oldest-posts"):
                return FakeResponse(status_code=204)
            raise AssertionError(f"Unexpected URL: {url}")

    context = fetch_tiktok_review_context(
        submitted_id,
        api_key="test-key",
        history_count=4,
        session=FakeSession(),
    )

    assert context["submission"]["views"] == 50_000
    assert context["submission"]["avg_views_30d"] == 10_000
    assert context["history_summary"]["oldest_posts_count"] == 0
    assert "history_warnings" not in context
