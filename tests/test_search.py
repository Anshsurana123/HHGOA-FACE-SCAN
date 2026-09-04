"""Unit tests for Stage 2: Search, URL filtering, and post extraction."""

import hashlib
import pytest

from search.post_extractor import (
    is_social_media_url,
    get_social_platform,
    normalize_url,
    _extract_opengraph_html,
    ALLOWED_DOMAINS,
)
from search.matcher import find_verified_social_post
import numpy as np


def test_social_platform_domain_filtering():
    """Validates allowed social media domain detection."""
    valid_urls = [
        ("https://x.com/user/status/123", "x"),
        ("https://twitter.com/user/status/123", "x"),
        ("https://www.instagram.com/p/C12345/", "instagram"),
        ("https://reddit.com/r/technology/comments/abc123/title/", "reddit"),
        ("https://www.threads.net/@user/post/123", "threads"),
        ("https://linkedin.com/posts/activity-12345", "linkedin"),
        ("https://pinterest.com/pin/12345/", "pinterest"),
        ("https://www.tiktok.com/@user/video/12345", "tiktok"),
        ("https://facebook.com/permalink.php?story_fbid=123", "facebook"),
    ]

    for url, expected_platform in valid_urls:
        assert is_social_media_url(url) is True, f"Failed for {url}"
        assert get_social_platform(url) == expected_platform, f"Platform mismatch for {url}"

    invalid_urls = [
        "https://example.com/article",
        "https://news.ycombinator.com/item?id=123",
        "https://wikipedia.org/wiki/Face",
        "https://randomblog.wordpress.com/post",
    ]

    for url in invalid_urls:
        assert is_social_media_url(url) is False, f"Should be invalid: {url}"
        assert get_social_platform(url) is None


def test_url_normalization():
    """Validates stripping of tracking parameters while preserving structure."""
    dirty_url = "https://x.com/user/status/12345?utm_source=twitter&utm_medium=social&s=20&t=abc"
    clean_url = normalize_url(dirty_url)
    assert clean_url == "https://x.com/user/status/12345"


def test_opengraph_extraction_parser():
    """Validates HTML meta and OpenGraph extraction parser."""
    html_sample = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta property="og:title" content="Breaking: Tech Announcement" />
        <meta property="og:description" content="A groundbreaking development announced today." />
        <meta property="og:image" content="https://images.example.com/photo.jpg" />
        <meta name="author" content="Jane Doe" />
        <meta property="article:published_time" content="2026-03-01T10:00:00Z" />
    </head>
    <body><h1>Hello World</h1></body>
    </html>
    """

    meta = _extract_opengraph_html(html_sample)
    assert "Tech Announcement" in meta["text"]
    assert "A groundbreaking development" in meta["text"]
    assert meta["image_url"] == "https://images.example.com/photo.jpg"
    assert meta["author"] == "Jane Doe"
    assert meta["posted_at"] == "2026-03-01T10:00:00Z"


def test_offline_demo_matcher():
    """Validates offline fallback mode for search matcher."""
    dummy_embedding = np.random.randn(512).astype(np.float32)
    result = find_verified_social_post(
        input_embedding=dummy_embedding,
        image_path_or_bytes="tests/fixtures/person1_a.jpg",
        tol=0.35,
        engine="yandex",
        max_candidates=50,
        offline_demo=True,
    )

    assert result.is_match_found is True
    assert result.accepted_record is not None
    assert result.accepted_record["platform"] == "x"
    assert "image_sha256" in result.accepted_record
    assert len(result.candidate_logs) == 1
    assert result.candidate_logs[0]["matched"] is True


def test_yandex_client_error_handling():
    """Validates Yandex client exception handling on missing key or failure."""
    from search.yandex_client import search_yandex_images, YandexSearchError
    with pytest.raises(YandexSearchError):
        search_yandex_images("https://invalid.example.com/test.jpg", api_key="")


def test_twitter_image_resolution_upgrade():
    """Validates upgrading Twitter CDN URLs from low-res thumbnails to original/400x400."""
    from search.post_extractor import _upgrade_twitter_image_url
    
    avatar_url = "https://pbs.twimg.com/profile_images/123456/photo_normal.jpg"
    upgraded_avatar = _upgrade_twitter_image_url(avatar_url)
    assert "photo_400x400.jpg" in upgraded_avatar

    media_url = "https://pbs.twimg.com/media/F123456?format=jpg&name=small"
    upgraded_media = _upgrade_twitter_image_url(media_url)
    assert "name=orig" in upgraded_media


def test_x_post_and_profile_live_extraction():
    """Validates live extraction of X post and profile metadata."""
    from search.post_extractor import fetch_post

    # Test real public tweet (Jack Dorsey tweet #20)
    jack_post = fetch_post("https://twitter.com/jack/status/20")
    assert jack_post["platform"] == "x"
    assert jack_post["author"] == "@jack"
    assert "just setting up my twttr" in jack_post["text"]
    assert jack_post["_image_url"] != ""
    assert jack_post["image_sha256"] != ""

    # Test real public profile (Barack Obama)
    obama_profile = fetch_post("https://x.com/BarackObama")
    assert obama_profile["platform"] == "x"
    assert obama_profile["author"] == "@BarackObama"
    assert "Barack Obama" in obama_profile["text"]
    assert obama_profile["_image_url"] != ""
    assert obama_profile["image_sha256"] != ""


def test_candidate_deduplication():
    """Validates that candidate lists are deduplicated across URLs and duplicate thumbnail URLs."""
    from search.matcher import _merge_candidate_lists

    cands1 = [
        {"link": "https://in.linkedin.com/in/user-one", "thumbnail": "https://cdn.example.com/photo1.jpg"},
        {"link": "https://www.linkedin.com/posts/activity-1", "thumbnail": "https://cdn.example.com/photo1.jpg"},  # Dup thumbnail
    ]
    cands2 = [
        {"link": "https://www.linkedin.com/in/user-one", "thumbnail": "https://cdn.example.com/photo1.jpg"},       # Dup canonical URL & thumb
        {"link": "https://x.com/user2/status/100", "thumbnail": "https://cdn.example.com/photo2.jpg"},             # Unique
    ]

    merged = _merge_candidate_lists(cands1, cands2)
    assert len(merged) == 2
    assert merged[0]["link"] == "https://in.linkedin.com/in/user-one"
    assert merged[1]["link"] == "https://x.com/user2/status/100"

