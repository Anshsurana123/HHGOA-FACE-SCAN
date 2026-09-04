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
    "fixupx.com",
    "fxtwitter.com",
    "vxtwitter.com",
    "xcancel.com",
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
    """Strips tracking parameters and normalizes regional subdomains & social mirrors."""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if ":" in netloc:
        netloc = netloc.split(":")[0]

    # Normalize regional LinkedIn (e.g. in.linkedin.com -> www.linkedin.com)
    if "linkedin.com" in netloc:
        netloc = "www.linkedin.com"

    # Normalize Pinterest regional subdomains (e.g. in.pinterest.com, ru.pinterest.com -> www.pinterest.com)
    if "pinterest." in netloc:
        netloc = "www.pinterest.com"

    # Normalize Twitter / X mirrors to canonical x.com
    if any(m in netloc for m in ("twitter.com", "twstalker.com", "nitter.net", "fixupx.com", "fxtwitter.com", "vxtwitter.com", "xcancel.com")):
        netloc = "x.com"

    allowed_params = []
    for k, v in parse_qsl(parsed.query):
        if not k.startswith("utm_") and k not in (
            "s", "t", "ref_src", "fbclid", "igshid", "trk", "originalSubdomain",
            "miniProfileUrn", "lipi", "licu", "midToken", "context", "share_id", "source"
        ):
            allowed_params.append((k, v))
    new_query = urlencode(allowed_params)
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
    return urlunparse((parsed.scheme or "https", netloc, path, parsed.params, new_query, ""))


def get_social_platform(url: str) -> str | None:
    """Returns the identified social platform name if URL belongs to an allowed social platform."""
    parsed = urlparse(url.lower())
    netloc = parsed.netloc
    if ":" in netloc:
        netloc = netloc.split(":")[0]

    if any(m in netloc for m in ("twstalker.com", "nitter.net", "fixupx.com", "fxtwitter.com", "vxtwitter.com", "xcancel.com")):
        return "x"

    if "pinterest." in netloc:
        return "pinterest"

    for domain in ALLOWED_DOMAINS:
        if netloc == domain or netloc.endswith("." + domain):
            name = domain.split(".")[0]
            return "x" if name in ("twitter", "twstalker", "nitter", "fixupx", "fxtwitter", "vxtwitter", "xcancel") else name
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
    """Fast HTTP GET with non-blocking timeouts and SSL verification fallback."""
    req_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if headers:
        req_headers.update(headers)

    t_tuple = (3.0, float(timeout) if timeout else 5.0)
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=req_headers, timeout=t_tuple)
            return resp
        except requests.exceptions.SSLError:
            try:
                import urllib3
                urllib3.disable_warnings()
                resp = requests.get(url, headers=req_headers, timeout=t_tuple, verify=False)
                return resp
            except requests.RequestException as exc:
                last_exc = exc
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


def _upgrade_twitter_image_url(image_url: str) -> str:
    """Upgrades Twitter/X image URLs to maximum resolution for biometric extraction."""
    if not image_url:
        return ""
    # Upgrade avatar thumbnails: _normal.jpg / _mini.jpg / _bigger.jpg -> _400x400.jpg
    if "pbs.twimg.com/profile_images/" in image_url:
        image_url = re.sub(r"_(?:normal|mini|bigger)\.", "_400x400.", image_url)
    # Upgrade tweet photos: ?name=small / ?name=900x900 -> ?name=orig
    elif "pbs.twimg.com/media/" in image_url:
        if "?" in image_url:
            image_url = re.sub(r"name=[a-zA-Z0-9]+", "name=orig", image_url)
        else:
            image_url = image_url + "?format=jpg&name=orig"
    return image_url


def _extract_x_post_or_profile(url: str) -> dict[str, Any] | None:
    """
    Extracts author, text, timestamp, and high-res photo from X / Twitter posts or profiles.
    Uses multi-tiered fallback:
      1. Direct JSON API via api.fxtwitter.com
      2. Direct JSON API via api.vxtwitter.com
      3. OpenGraph crawler via fxtwitter.com / fixupx.com
      4. Twitter oEmbed endpoint
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    parts = [p for p in path.split("/") if p]
    if not parts:
        return None

    # Strategy A: api.fxtwitter.com
    for api_host in ("https://api.fxtwitter.com", "https://api.vxtwitter.com"):
        try:
            api_endpoint = f"{api_host}/{path}"
            resp = _http_get(api_endpoint, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                if "tweet" in data and data["tweet"]:
                    tw = data["tweet"]
                    screen_name = tw.get("author", {}).get("screen_name", "")
                    author = f"@{screen_name}" if screen_name else tw.get("author", {}).get("name", "")
                    text = tw.get("text") or tw.get("raw_text", {}).get("text", "")
                    posted_at = tw.get("created_at") or (str(tw.get("created_timestamp")) if tw.get("created_timestamp") else "")
                    
                    # Extract high-res image (photos or video thumbnail, never raw mp4)
                    media_photos = tw.get("media", {}).get("photos", [])
                    media_videos = tw.get("media", {}).get("videos", [])
                    media_all = tw.get("media", {}).get("all", [])
                    img_url = ""
                    if media_photos and isinstance(media_photos, list):
                        img_url = media_photos[0].get("url", "")
                    elif media_videos and isinstance(media_videos, list):
                        img_url = media_videos[0].get("thumbnail_url") or media_videos[0].get("thumbnail") or ""
                    elif media_all and isinstance(media_all, list):
                        m0 = media_all[0]
                        if m0.get("type") in ("video", "gif") or ".mp4" in m0.get("url", "").lower():
                            img_url = m0.get("thumbnail_url") or m0.get("thumbnail") or ""
                        else:
                            img_url = m0.get("url", "")

                    if not img_url or ".mp4" in img_url.lower():
                        img_url = tw.get("author", {}).get("avatar_url", "")

                    return {
                        "author": author,
                        "text": text,
                        "posted_at": posted_at,
                        "image_url": _upgrade_twitter_image_url(img_url),
                    }
                elif "user" in data and data["user"]:
                    u = data["user"]
                    screen_name = u.get("screen_name", "")
                    author = f"@{screen_name}" if screen_name else u.get("name", "")
                    name = u.get("name", "")
                    desc = u.get("description", "")
                    text = f"{name} - {desc}".strip(" -") if (name and desc) else (desc or name)
                    posted_at = u.get("joined", "")
                    img_url = u.get("avatar_url", "")

                    return {
                        "author": author,
                        "text": text,
                        "posted_at": posted_at,
                        "image_url": _upgrade_twitter_image_url(img_url),
                    }
        except Exception:
            pass

    # Strategy B: OpenGraph crawl on fxtwitter.com
    try:
        fxtw_url = f"https://fxtwitter.com/{path}"
        resp = _http_get(fxtw_url, headers={"User-Agent": UA_TWITTER_BOT}, timeout=6)
        if resp.status_code == 200:
            og_meta = _extract_opengraph_html(resp.text)
            if og_meta.get("text") or og_meta.get("image_url"):
                og_meta["image_url"] = _upgrade_twitter_image_url(og_meta.get("image_url", ""))
                return og_meta
    except Exception:
        pass

    # Strategy C: Twitter oEmbed fallback
    return _extract_twitter_oembed(url)


def _extract_twitter_oembed(url: str) -> dict[str, Any] | None:
    """Attempts to fetch metadata via Twitter/X oEmbed endpoint."""
    oembed_url = f"https://publish.twitter.com/oembed?url={url}&omit_script=true"
    try:
        resp = _http_get(oembed_url, timeout=5)
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
    """
    Extracts post image, author, and caption from Instagram public embed endpoint.
    Filters out 1x1 data URI placeholders and parses high-res CDN images from HTML and embedded state.
    """
    match = re.search(r"/(?:p|reel|tv)/([^/?#&]+)", url)
    if not match:
        return None
    shortcode = match.group(1)
    embed_url = f"https://www.instagram.com/p/{shortcode}/embed/captioned/"
    try:
        resp = _http_get(embed_url, headers={"User-Agent": UA_DESKTOP_CHROME}, timeout=6)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Extract author
            username_tag = soup.find("a", class_="CaptionUsername")
            author = f"@{username_tag.get_text().strip()}" if username_tag else ""

            # Extract caption text
            caption_tag = soup.find("div", class_="Caption")
            text = caption_tag.get_text(separator=" ").strip() if caption_tag else ""

            # Extract genuine image URL (ignoring data:image base64 placeholders)
            image_url = ""
            for img in soup.find_all("img"):
                src = img.get("src", "")
                if src and not src.startswith("data:") and ("cdninstagram" in src or "fbcdn" in src or "instagram" in src):
                    image_url = src
                    break

            # If not in img tag, check embedded scripts for CDN URLs
            if not image_url:
                for s in soup.find_all("script"):
                    script_text = s.string or ""
                    cdn_matches = re.findall(r'https://[^\s"\'<>\\]+cdninstagram\.com[^\s"\'<>\\]+', script_text)
                    if cdn_matches:
                        # Clean backslash escapes in JSON
                        image_url = cdn_matches[0].replace(r"\/", "/")
                        break

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

    # If author is missing, extract from title or og:url (e.g. Instagram handle)
    if not author:
        tw_title = get_meta("twitter:title")
        if "(@" in tw_title:
            m = re.search(r"\(@([a-zA-Z0-9._]+)\)", tw_title)
            if m:
                author = f"@{m.group(1)}"
        if not author:
            og_url = get_meta("og:url")
            m = re.search(r"instagram\.com/([a-zA-Z0-9._]+)/(?:p|reel|tv)/", og_url)
            if m:
                author = f"@{m.group(1)}"

    # If posted_at is missing, attempt to extract human date from description
    if not posted_at:
        desc = get_meta("description", "og:description")
        m = re.search(r"on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", desc)
        if m:
            posted_at = m.group(1)

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
        x_meta = _extract_x_post_or_profile(canonical_url)
        if x_meta:
            author = x_meta.get("author", "")
            text = x_meta.get("text", "")
            posted_at = x_meta.get("posted_at", "")
            image_url = x_meta.get("image_url", "")

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
    # If image_url is an SVG or missing, attempt to find a raster image in page HTML
    raw_html_cache = ""
    if (not image_url or image_url.lower().endswith(".svg") or ".svg?" in image_url.lower()) and platform not in ("x", "twitter"):
        try:
            resp = _http_get(canonical_url, headers={"User-Agent": UA_DESKTOP_CHROME}, timeout=timeout)
            if resp.status_code == 200:
                raw_html_cache = resp.text
                soup = BeautifulSoup(raw_html_cache, "html.parser")
                for img_tag in soup.find_all("img"):
                    src = img_tag.get("src", "").strip()
                    if src and not src.lower().endswith(".svg") and ".svg" not in src.lower() and not src.startswith("data:"):
                        image_url = src
                        break
        except Exception:
            pass

    if image_url and not any(v in image_url.lower() for v in (".mp4", ".mov", ".m3u8", ".webm", ".avi", "/vid/")):
        try:
            if image_url.startswith("//"):
                image_url = "https:" + image_url
            elif image_url.startswith("/"):
                parsed = urlparse(canonical_url)
                image_url = f"{parsed.scheme}://{parsed.netloc}{image_url}"

            img_headers = {
                "User-Agent": UA_DESKTOP_CHROME,
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Referer": canonical_url,
            }
            img_resp = _http_get(image_url, headers=img_headers, timeout=timeout)
            if img_resp.status_code != 200 or len(img_resp.content) == 0:
                # Retry with bot UA
                img_resp = _http_get(image_url, headers={"User-Agent": UA_FACEBOOK_BOT}, timeout=timeout)

            if img_resp.status_code == 200 and len(img_resp.content) > 0:
                # Check if this format is readable by cv2 or PIL
                try:
                    import numpy as np
                    import cv2
                    nparr = np.frombuffer(img_resp.content, np.uint8)
                    t_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if t_img is not None:
                        image_bytes = img_resp.content
                        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
                    else:
                        from PIL import Image
                        import io
                        _ = Image.open(io.BytesIO(img_resp.content))
                        image_bytes = img_resp.content
                        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
                except Exception:
                    image_bytes = None
                    image_sha256 = ""
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
