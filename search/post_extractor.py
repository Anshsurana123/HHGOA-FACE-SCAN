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

# Specialized OpenGraph Link Unfurl Crawlers & Standard Browser UA
UA_FACEBOOK_BOT = "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)"
UA_WHATSAPP_BOT = "WhatsApp/2.21.12.21 A"
UA_TWITTER_BOT = "Twitterbot/1.0"
UA_LINKEDIN_BOT = "LinkedInBot/1.0 (compatible; Mozilla/5.0; Apache-HttpClient +http://www.linkedin.com)"
UA_DESKTOP_CHROME = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

USER_AGENT = UA_DESKTOP_CHROME


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


def _get_platform_user_agent(platform: str) -> str:
    """Returns the most effective link-preview crawler User-Agent for the specific platform."""
    if platform in ("instagram", "facebook", "threads"):
        return UA_FACEBOOK_BOT
    elif platform in ("x", "twitter"):
        return UA_TWITTER_BOT
    elif platform == "linkedin":
        return UA_LINKEDIN_BOT
    return UA_DESKTOP_CHROME


def _extract_author_from_url(url: str, platform: str) -> str:
    """Extracts the username/handle from social URL path when page metadata lacks author."""
    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        parts = [p for p in path.split("/") if p]
        if not parts:
            return ""

        if platform in ("instagram", "threads"):
            # Format: instagram.com/username/ or instagram.com/p/shortcode/
            if parts[0] in ("p", "reel", "reels", "tv", "stories"):
                return ""
            return f"@{parts[0]}" if not parts[0].startswith("@") else parts[0]

        elif platform in ("x", "twitter"):
            # Format: x.com/username/status/... or x.com/username
            if parts[0] not in ("i", "hashtag", "search", "explore"):
                return f"@{parts[0]}" if not parts[0].startswith("@") else parts[0]

        elif platform == "facebook":
            # Format: facebook.com/username or facebook.com/profile.php?id=...
            if parts[0] not in ("photo", "photos", "posts", "watch", "permalink.php", "story.php", "share", "groups"):
                return parts[0]

        elif platform == "reddit":
            # Format: reddit.com/r/subreddit/... or reddit.com/u/user/...
            if len(parts) >= 2 and parts[0] in ("r", "u", "user"):
                prefix = "r" if parts[0] == "r" else "u"
                return f"{prefix}/{parts[1]}"

        elif platform == "tiktok":
            if parts[0].startswith("@"):
                return parts[0]

        elif platform == "pinterest":
            if parts[0] not in ("pin", "search", "ideas"):
                return f"@{parts[0]}"
    except Exception:
        pass
    return ""


def _http_get(url: str, headers: dict | None = None, timeout: int = 5, max_retries: int = 0) -> requests.Response:
    """Fast HTTP GET with non-blocking timeouts."""
    req_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
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
        resp = _http_get(json_url, headers={"User-Agent": f"HH-FaceChain/1.0 ({UA_DESKTOP_CHROME})"})
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            post_data = data[0].get("data", {}).get("children", [{}])[0].get("data", {})
            author = post_data.get("author", "")
            if author:
                author = f"u/{author}"
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
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator=" ").strip()
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


def _extract_instagram_embed(url: str) -> dict[str, Any] | None:
    """Attempts to extract post image & caption from Instagram public embed endpoint."""
    match = re.search(r"/(?:p|reel|tv)/([^/?#&]+)", url)
    if not match:
        return None
    shortcode = match.group(1)
    embed_url = f"https://www.instagram.com/p/{shortcode}/embed/captioned/"
    try:
        resp = _http_get(embed_url, headers={"User-Agent": UA_DESKTOP_CHROME}, timeout=5)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            img_tag = soup.find("img", class_="EmbeddedMediaImage") or soup.find("img")
            image_url = img_tag["src"] if (img_tag and img_tag.get("src")) else ""
            caption_tag = soup.find("div", class_="Caption")
            text = caption_tag.get_text(separator=" ").strip() if caption_tag else ""
            username_tag = soup.find("a", class_="CaptionUsername")
            author = f"@{username_tag.get_text().strip()}" if username_tag else ""
            return {
                "author": author,
                "text": text,
                "posted_at": "",
                "image_url": image_url,
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

    # Strategy 1: Specialized platform endpoints
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

    elif platform == "instagram":
        # Check Instagram embed if post has a shortcode
        ig_embed = _extract_instagram_embed(canonical_url)
        if ig_embed:
            author = ig_embed.get("author", "")
            text = ig_embed.get("text", "")
            image_url = ig_embed.get("image_url", "")

    # Strategy 2: OpenGraph crawler requests with platform-specific bot User-Agents
    if not text or not image_url:
        crawler_ua = _get_platform_user_agent(platform)
        try:
            resp = _http_get(canonical_url, headers={"User-Agent": crawler_ua}, timeout=timeout)
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

    # Strategy 2b: Fallback to WhatsApp Bot UA if Facebook Bot didn't retrieve image for Meta
    if not image_url and platform in ("instagram", "facebook", "threads"):
        try:
            resp = _http_get(canonical_url, headers={"User-Agent": UA_WHATSAPP_BOT}, timeout=timeout)
            if resp.status_code == 200:
                og_meta = _extract_opengraph_html(resp.text)
                if not image_url:
                    image_url = og_meta.get("image_url", "")
                if not text:
                    text = og_meta.get("text", "")
        except Exception:
            pass

    # Author fallback from URL structure
    if not author:
        author = _extract_author_from_url(canonical_url, platform)

    # Strategy 3: Download Post Image bytes & compute SHA-256
    if image_url:
        try:
            if image_url.startswith("//"):
                image_url = "https:" + image_url
            elif image_url.startswith("/"):
                parsed = urlparse(canonical_url)
                image_url = f"{parsed.scheme}://{parsed.netloc}{image_url}"

            img_headers = {
                "User-Agent": UA_DESKTOP_CHROME,
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                "Referer": canonical_url,
            }
            img_resp = _http_get(image_url, headers=img_headers, timeout=timeout)
            if img_resp.status_code != 200 or len(img_resp.content) == 0:
                # Retry with bot UA
                img_resp = _http_get(image_url, headers={"User-Agent": UA_FACEBOOK_BOT}, timeout=timeout)

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
