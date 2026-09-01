"""SerpApi Yandex Images client for deep facial reverse image search."""

from __future__ import annotations

import os
import time
from typing import Any
import requests


class YandexSearchError(Exception):
    """Raised when reverse image search via SerpApi Yandex Images fails."""
    pass


DEFAULT_USER_AGENT = "HH-FaceChain/1.0 (Yandex Visual Search Bot; +https://github.com)"


def search_yandex_images(
    image_url: str,
    api_key: str | None = None,
    timeout: int = 20,
    max_retries: int = 1,
) -> list[dict[str, Any]]:
    """
    Queries SerpApi Yandex Images engine (industry gold standard for facial reverse search).

    Args:
        image_url: Publicly accessible URL of the face scan.
        api_key: SerpApi key (defaults to SERPAPI_KEY env var).
        timeout: Request timeout in seconds.
        max_retries: Retry attempts on network/server failures.

    Returns:
        list[dict]: List of match result dicts with fields:
                    `title`, `link`, `source`, `thumbnail`, `original_image`, `position`.

    Raises:
        YandexSearchError: If API key is missing or query fails.
    """
    key = api_key or os.getenv("SERPAPI_KEY")
    if not key:
        raise YandexSearchError("Missing SerpApi key. Set SERPAPI_KEY environment variable.")

    endpoint = "https://serpapi.com/search"
    params = {
        "engine": "yandex_images",
        "url": image_url,
        "api_key": key,
    }
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    raise YandexSearchError(f"SerpApi Yandex error: {data['error']}")

                candidates: list[dict[str, Any]] = []

                # 1. Parse image_results (contains exact pages featuring this face)
                image_results = data.get("image_results", [])
                for item in image_results:
                    link = item.get("link")
                    if not link:
                        continue

                    # Extract thumbnail URL string
                    thumb_val = item.get("thumbnail")
                    thumb_url = ""
                    if isinstance(thumb_val, dict):
                        thumb_url = thumb_val.get("link", "")
                    elif isinstance(thumb_val, str):
                        thumb_url = thumb_val

                    # Extract original source image URL if provided
                    orig_val = item.get("original_image")
                    orig_url = ""
                    if isinstance(orig_val, dict):
                        orig_url = orig_val.get("link", "")
                    elif isinstance(orig_val, str):
                        orig_url = orig_val

                    title = item.get("title") or item.get("snippet") or "Web Match"

                    candidates.append({
                        "title": title,
                        "link": link,
                        "source": item.get("source", ""),
                        "thumbnail": thumb_url,
                        "original_image": orig_url,
                        "position": len(candidates) + 1,
                    })

                # 2. Parse similar_images if image_results was empty or sparse
                if len(candidates) < 5:
                    similar_images = data.get("similar_images", [])
                    for item in similar_images:
                        link = item.get("link")
                        if not link or any(c["link"] == link for c in candidates):
                            continue

                        thumb_val = item.get("thumbnail")
                        thumb_url = thumb_val.get("link", "") if isinstance(thumb_val, dict) else (thumb_val or "")

                        orig_val = item.get("original_image")
                        orig_url = orig_val.get("link", "") if isinstance(orig_val, dict) else (orig_val or "")

                        candidates.append({
                            "title": item.get("title") or item.get("snippet") or "Similar Match",
                            "link": link,
                            "source": item.get("source", ""),
                            "thumbnail": thumb_url,
                            "original_image": orig_url,
                            "position": len(candidates) + 1,
                        })

                return candidates

            error_msg = response.text
            try:
                err_json = response.json()
                error_msg = err_json.get("error", error_msg)
            except Exception:
                pass

            raise YandexSearchError(f"HTTP {response.status_code}: {error_msg}")

        except (requests.RequestException, YandexSearchError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))

    raise YandexSearchError(f"Yandex search failed after {max_retries + 1} attempt(s): {last_error}")
