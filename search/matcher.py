"""Search and face match orchestrator for social media posts."""

from __future__ import annotations

import os
import time
from typing import Any
import numpy as np

from faceid.encoder import (
    encode_face,
    faces_match,
    cosine_distance,
    NoFaceFound,
    ImageReadError,
)
from search.imgbb_client import upload_to_imgbb
from search.lens_client import search_google_lens
from search.post_extractor import (
    fetch_post,
    is_social_media_url,
    normalize_url,
    get_social_platform,
    _http_get,
)


class MatcherResult:
    """Encapsulates the result of the search and face matching process."""

    def __init__(
        self,
        accepted_record: dict[str, Any] | None,
        candidate_logs: list[dict[str, Any]],
        imgbb_url: str | None = None,
        total_lens_matches: int = 0,
        total_social_candidates: int = 0,
        accepted_distance: float | None = None,
        reason: str = "",
    ):
        self.accepted_record = accepted_record
        self.candidate_logs = candidate_logs
        self.imgbb_url = imgbb_url
        self.total_lens_matches = total_lens_matches
        self.total_social_candidates = total_social_candidates
        self.accepted_distance = accepted_distance
        self.reason = reason

    @property
    def is_match_found(self) -> bool:
        return self.accepted_record is not None


def find_verified_social_post(
    input_embedding: np.ndarray,
    image_path_or_bytes: str | bytes,
    tol: float = 0.35,
    serpapi_key: str | None = None,
    imgbb_key: str | None = None,
    max_candidates: int = 5,
    offline_demo: bool = False,
    offline_post_url: str | None = None,
) -> MatcherResult:
    """
    Executes Stage 2:
    1. Upload scan to ImgBB -> public URL
    2. Query Google Lens via SerpApi
    3. Filter to social media platforms
    4. Fetch top candidates, download image, encode face, compare cosine distance < tol
    5. Return first verified match.
    """
    candidate_logs: list[dict[str, Any]] = []

    # Offline demo fallback path
    if offline_demo:
        demo_url = offline_post_url or "https://x.com/demo_user/status/1234567890"
        post_data = {
            "platform": "x",
            "post_url": demo_url,
            "author": "Demo Public Figure",
            "text": "Verified sample post for offline pipeline validation.",
            "posted_at": "2026-01-01T12:00:00Z",
            "image_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "_image_bytes": b"demo_image_bytes",
        }
        candidate_logs.append({
            "position": 1,
            "url": demo_url,
            "platform": "x",
            "distance": 0.12,
            "matched": True,
            "status": "ACCEPTED (Offline Demo Mode)",
        })
        return MatcherResult(
            accepted_record=post_data,
            candidate_logs=candidate_logs,
            imgbb_url="https://i.ibb.co/demo/scan.jpg",
            total_lens_matches=1,
            total_social_candidates=1,
            accepted_distance=0.12,
            reason="Matched via offline demo mode.",
        )

    # Step 1: Upload image to ImgBB
    imgbb_url = upload_to_imgbb(image_path_or_bytes, api_key=imgbb_key)

    # Step 2: Query Google Lens
    lens_results = search_google_lens(imgbb_url, api_key=serpapi_key)
    total_lens_matches = len(lens_results)

    if not lens_results:
        return MatcherResult(
            accepted_record=None,
            candidate_logs=candidate_logs,
            imgbb_url=imgbb_url,
            total_lens_matches=0,
            total_social_candidates=0,
            reason="No visual matches returned by Google Lens for this image.",
        )

    # Step 3: Filter result links to social media platforms
    social_candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for item in lens_results:
        link = item.get("link", "")
        if not link:
            continue
        canon_link = normalize_url(link)
        if is_social_media_url(canon_link) and canon_link not in seen_urls:
            seen_urls.add(canon_link)
            social_candidates.append(item)

    total_social_candidates = len(social_candidates)
    if not social_candidates:
        return MatcherResult(
            accepted_record=None,
            candidate_logs=candidate_logs,
            imgbb_url=imgbb_url,
            total_lens_matches=total_lens_matches,
            total_social_candidates=0,
            reason="Google Lens returned matches, but none belonged to supported social media platforms.",
        )

    # Step 4 & 5: Evaluate top candidates in rank order
    candidates_to_eval = social_candidates[:max_candidates]

    for idx, candidate in enumerate(candidates_to_eval, start=1):
        url = candidate.get("link", "")
        platform = get_social_platform(url) or "social"
        candidate_log: dict[str, Any] = {
            "position": idx,
            "url": url,
            "platform": platform,
            "title": candidate.get("title", ""),
            "distance": None,
            "matched": False,
            "status": "PROCESSING",
        }

        try:
            # Fetch post metadata using shared function
            post_data = fetch_post(url)

            # Look for face in post image or candidate thumbnail
            img_bytes_to_check = post_data.get("_image_bytes")

            # Fallback to candidate thumbnail for face verification if direct og:image wasn't available
            if not img_bytes_to_check and candidate.get("thumbnail"):
                try:
                    thumb_resp = _http_get(candidate["thumbnail"], timeout=10)
                    if thumb_resp.status_code == 200 and len(thumb_resp.content) > 0:
                        img_bytes_to_check = thumb_resp.content
                        post_data["_image_bytes"] = img_bytes_to_check
                except Exception:
                    pass

            if not img_bytes_to_check:
                candidate_log["status"] = "REJECTED: No image could be retrieved from post."
                candidate_logs.append(candidate_log)
                continue

            # Detect face in the retrieved post image
            try:
                post_embedding = encode_face(img_bytes_to_check)
            except NoFaceFound:
                candidate_log["status"] = "REJECTED: No face detected in post image."
                candidate_logs.append(candidate_log)
                continue
            except ImageReadError:
                candidate_log["status"] = "REJECTED: Post image bytes could not be decoded."
                candidate_logs.append(candidate_log)
                continue

            # Compare embeddings
            dist = cosine_distance(input_embedding, post_embedding)
            candidate_log["distance"] = round(dist, 4)

            if dist < tol:
                candidate_log["matched"] = True
                candidate_log["status"] = f"ACCEPTED (cosine distance: {dist:.4f} < {tol})"
                candidate_logs.append(candidate_log)

                return MatcherResult(
                    accepted_record=post_data,
                    candidate_logs=candidate_logs,
                    imgbb_url=imgbb_url,
                    total_lens_matches=total_lens_matches,
                    total_social_candidates=total_social_candidates,
                    accepted_distance=dist,
                    reason=f"Verified match found on {platform} (distance: {dist:.4f}).",
                )
            else:
                candidate_log["status"] = f"REJECTED: Cosine distance {dist:.4f} >= {tol} (different person)."
                candidate_logs.append(candidate_log)

        except Exception as exc:
            candidate_log["status"] = f"ERROR: Failed during extraction/verification: {exc}"
            candidate_logs.append(candidate_log)

    return MatcherResult(
        accepted_record=None,
        candidate_logs=candidate_logs,
        imgbb_url=imgbb_url,
        total_lens_matches=total_lens_matches,
        total_social_candidates=total_social_candidates,
        reason=f"Evaluated {len(candidates_to_eval)} candidate(s), but none satisfied the face match tolerance (tol={tol}).",
    )
