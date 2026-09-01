"""Social media URL filtering, metadata extraction, and post fetching."""

from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Any
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import requests
from bs4 import BeautifulSoup


class PostExtractionError(Exception):
    """Raised when post extraction or network fetch fails."""
    pass


ALLOWED_DOMAINS = [
    "x.com",
    "twitter.com",
    "instagram.com",
    "facebook.com",
    "reddit.com",
    "linkedin.com",
    "threads.net",
    "pinterest.com",
    "tiktok.com",
    "twstalker.com",
    "nitter.net",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def normalize_url(url: str) -> str:
    """Strips tracking parameters and normalizes regional subdomains (e.g. in.linkedin.com -> www.linkedin.com)."""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if ":" in netloc:
        netloc = netloc.split(":")[0]

    # Normalize regional LinkedIn (e.g. in.linkedin.com -> www.linkedin.com)
    if "linkedin.com" in netloc:
        netloc = "www.linkedin.com"

    # Normalize Twitter mirrors (e.g. twstalker.com/Username -> x.com/Username)
    if "twstalker.com" in netloc or "nitter.net" in netloc:
        path_parts = [p for p in parsed.path.split("/") if p]
        if path_parts:
            username = path_parts[0]
            return f"https://x.com/{username}"

    allowed_params = []
    for k, v in parse_qsl(parsed.query):
        if not k.startswith("utm_") and k not in ("s", "t", "ref_src", "fbclid", "igshid", "trk", "originalSubdomain"):
            allowed_params.append((k, v))
    new_query = urlencode(allowed_params)
    return urlunparse((parsed.scheme or "https", netloc, parsed.path, parsed.params, new_query, ""))


def get_social_platform(url: str) -> str | None:
    """Returns the identified social platform name if URL belongs to an allowed social platform."""
    parsed = urlparse(url.lower())
    netloc = parsed.netloc
    if ":" in netloc:
        netloc = netloc.split(":")[0]

    if "twstalker.com" in netloc or "nitter.net" in netloc:
        return "x"

    for domain in ALLOWED_DOMAINS:
        if netloc == domain or netloc.endswith("." + domain):
            name = domain.split(".")[0]
            return "x" if name in ("twitter", "twstalker", "nitter") else name
    return None


def is_social_media_url(url: str) -> bool:
    """Checks whether the given URL is hosted on an allowed social media platform."""
    return get_social_platform(url) is not None


def _http_get(url: str, headers: dict | None = None, timeout: int = 4, max_retries: int = 0) -> requests.Response:
    """Fast HTTP GET with non-blocking timeouts."""
    req_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        req_headers.update(headers)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=req_headers, timeout=timeout)
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(0.3)
                continue
            raise PostExtractionError(f"HTTP request to {url} failed: {exc}") from exc

    raise PostExtractionError(f"HTTP request to {url} failed: {last_exc}")


def _extract_reddit_json(url: str) -> dict[str, Any] | None:
    """Attempts to fetch metadata from Reddit's public .json endpoint."""
    json_url = url.rstrip("/") + ".json"
    try:
        resp = _http_get(json_url, headers={"User-Agent": USER_AGENT + " (HH-FaceChain)"})
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            post_data = data[0].get("data", {}).get("children", [{}])[0].get("data", {})
            author = post_data.get("author", "")
            title = post_data.get("title", "")
            selftext = post_data.get("selftext", "")
            text = f"{title}\n{selftext}".strip() if selftext else title
            created_utc = post_data.get("created_utc")
            posted_at = str(int(created_utc)) if created_utc else ""

            image_url = ""
            url_overridden = post_data.get("url_overridden_by_dest", "")
            if any(url_overridden.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                image_url = url_overridden
            elif "preview" in post_data and "images" in post_data["preview"] and len(post_data["preview"]["images"]) > 0:
                image_url = post_data["preview"]["images"][0].get("source", {}).get("url", "")
                image_url = image_url.replace("&amp;", "&")

            return {
                "author": author,
                "text": text,
                "posted_at": posted_at,
                "image_url": image_url,
            }
    except Exception:
        pass
    return None


def _extract_twitter_oembed(url: str) -> dict[str, Any] | None:
    """Attempts to fetch metadata via Twitter/X oEmbed endpoint."""
    oembed_url = f"https://publish.twitter.com/oembed?url={url}&omit_script=true"
    try:
        resp = _http_get(oembed_url)
        if resp.status_code == 200:
            data = resp.json()
            author = data.get("author_name", "")
            html = data.get("html", "")
            # Clean HTML tags to obtain post text
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator=" ").strip()
            # Remove any trailing date link text added by oEmbed
            text = re.sub(r"—\s*.*$", "", text, flags=re.MULTILINE).strip()
            return {
                "author": author,
                "text": text,
                "posted_at": "",
                "image_url": "",
            }
    except Exception:
        pass
    return None


def _extract_opengraph_html(html_text: str) -> dict[str, Any]:
    """Extracts OpenGraph and standard HTML meta tags using BeautifulSoup."""
    soup = BeautifulSoup(html_text, "html.parser")

    def get_meta(*names: str) -> str:
        for name in names:
            tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return ""

    title = get_meta("og:title", "twitter:title")
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    description = get_meta("og:description", "twitter:description", "description")
    text = f"{title}\n{description}".strip() if (title and description and title != description) else (description or title)

    image_url = get_meta("og:image", "twitter:image", "twitter:image:src")
    if not image_url:
        link_tag = soup.find("link", attrs={"rel": "image_src"})
        if link_tag and link_tag.get("href"):
            image_url = link_tag["href"].strip()

    author = get_meta("author", "article:author", "twitter:creator", "og:article:author")
    posted_at = get_meta("article:published_time", "og:updated_time", "datePublished")

    return {
        "author": author,
        "text": text,
        "posted_at": posted_at,
        "image_url": image_url,
    }


def fetch_post(url: str, timeout: int = 10) -> dict[str, Any]:
    """
    Fetches and extracts metadata and post image from a given post URL.
    This exact function is shared between anchor-time and live verify-time.

    Returns dict with keys:
        - platform (str)
        - post_url (str)
        - author (str)
        - text (str)
        - posted_at (str)
        - image_sha256 (str)
        - _image_bytes (bytes | None)
        - _image_url (str)
    """
    platform = get_social_platform(url) or "web"
    canonical_url = normalize_url(url)

    author = ""
    text = ""
    posted_at = ""
    image_url = ""
    image_bytes: bytes | None = None
    image_sha256 = ""

    # Strategy 1: Specialized oEmbed / JSON endpoints
    if platform == "reddit":
        reddit_meta = _extract_reddit_json(canonical_url)
        if reddit_meta:
            author = reddit_meta.get("author", "")
            text = reddit_meta.get("text", "")
            posted_at = reddit_meta.get("posted_at", "")
            image_url = reddit_meta.get("image_url", "")

    elif platform in ("x", "twitter"):
        tw_meta = _extract_twitter_oembed(canonical_url)
        if tw_meta:
            author = tw_meta.get("author", "")
            text = tw_meta.get("text", "")

    # Strategy 2: HTML OpenGraph and Meta tags
    if not text or not image_url:
        try:
            resp = _http_get(canonical_url, timeout=timeout)
            if resp.status_code == 200:
                og_meta = _extract_opengraph_html(resp.text)
                if not author:
                    author = og_meta.get("author", "")
                if not text:
                    text = og_meta.get("text", "")
                if not posted_at:
                    posted_at = og_meta.get("posted_at", "")
                if not image_url:
                    image_url = og_meta.get("image_url", "")
        except Exception:
            pass

    # Strategy 3: Download Post Image bytes & compute SHA-256
    if image_url:
        try:
            # Resolve relative URLs
            if image_url.startswith("//"):
                image_url = "https:" + image_url
            elif image_url.startswith("/"):
                parsed = urlparse(canonical_url)
                image_url = f"{parsed.scheme}://{parsed.netloc}{image_url}"

            img_resp = _http_get(image_url, timeout=timeout)
            if img_resp.status_code == 200 and len(img_resp.content) > 0:
                image_bytes = img_resp.content
                image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        except Exception:
            image_bytes = None
            image_sha256 = ""

    return {
        "platform": platform,
        "post_url": canonical_url,
        "author": author,
        "text": text,
        "posted_at": posted_at,
        "image_sha256": image_sha256,
        "_image_bytes": image_bytes,
        "_image_url": image_url,
    }
