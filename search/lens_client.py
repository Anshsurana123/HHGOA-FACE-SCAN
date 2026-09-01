"""SerpApi Google Lens client for reverse image searching."""

from __future__ import annotations

import os
import time
from typing import Any
import requests


class LensSearchError(Exception):
    """Raised when reverse image search via SerpApi Google Lens fails."""
    pass


DEFAULT_USER_AGENT = "HH-FaceChain/1.0 (Reverse Image Search Bot; +https://github.com)"


def search_google_lens(
    image_url: str,
    api_key: str | None = None,
    timeout: int = 15,
    max_retries: int = 1,
) -> list[dict[str, Any]]:
    """
    Queries SerpApi Google Lens engine with a public image URL.

    Args:
        image_url: Publicly accessible URL of the face scan.
        api_key: SerpApi key (defaults to SERPAPI_KEY env var).
        timeout: Request timeout in seconds.
        max_retries: Retry attempts on network/server failures.

    Returns:
        list[dict]: List of match result dicts with fields:
                    `title`, `link`, `source`, `thumbnail`, `position`.

    Raises:
        LensSearchError: If API key is missing or query fails.
    """
    key = api_key or os.getenv("SERPAPI_KEY")
    if not key:
        raise LensSearchError("Missing SerpApi key. Set SERPAPI_KEY environment variable.")

    endpoint = "https://serpapi.com/search"
    params = {
        "engine": "google_lens",
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
                    raise LensSearchError(f"SerpApi error: {data['error']}")

                candidates: list[dict[str, Any]] = []

                # 1. Parse visual_matches
                visual_matches = data.get("visual_matches", [])
                for item in visual_matches:
                    link = item.get("link")
                    if link:
                        candidates.append({
                            "title": item.get("title", ""),
                            "link": link,
                            "source": item.get("source", ""),
                            "thumbnail": item.get("thumbnail", ""),
                            "position": item.get("position", len(candidates) + 1),
                        })

                # 2. Parse knowledge_graph / exact_matches if present
                knowledge_graph = data.get("knowledge_graph", [])
                if isinstance(knowledge_graph, list):
                    for item in knowledge_graph:
                        link = item.get("link")
                        if link and not any(c["link"] == link for c in candidates):
                            candidates.append({
                                "title": item.get("title", ""),
                                "link": link,
                                "source": item.get("source", ""),
                                "thumbnail": item.get("thumbnail", ""),
                                "position": len(candidates) + 1,
                            })

                return candidates

            error_msg = response.text
            try:
                err_json = response.json()
                if "error" in err_json:
                    error_msg = err_json["error"]
            except Exception:
                pass
            raise LensSearchError(f"SerpApi HTTP {response.status_code}: {error_msg}")

        except (requests.RequestException, LensSearchError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.0)
                continue
            raise LensSearchError(f"Google Lens search failed after {max_retries + 1} attempt(s): {exc}") from exc

    raise LensSearchError(f"Google Lens search failed: {last_error}")
