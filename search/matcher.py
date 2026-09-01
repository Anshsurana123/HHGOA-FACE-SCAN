"""Search and face match orchestrator for social media posts.

Supports:
  - Yandex Images (Primary / Default for deep facial identification)
  - Google Lens (Secondary / Fallback)
  - Hybrid Multi-Engine (Combines Yandex + Google Lens)
  - Tiered evidence strength with perceptual aHash near-duplicate detection
  - Full candidate pool evaluation with configurable search depth
"""

from __future__ import annotations

import io
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
from search.yandex_client import search_yandex_images
from search.post_extractor import (
    fetch_post,
    is_social_media_url,
    normalize_url,
    get_social_platform,
    _http_get,
)


# Evidence-strength tiers, best first.
TIER_PAGE_IMAGE_UNIQUE = 0       # downloaded from real post/page, distinct photo
TIER_PAGE_IMAGE_DUPLICATE = 1    # downloaded from real post/page, but duplicate of input
TIER_THUMBNAIL_UNIQUE = 2        # only engine thumbnail was available, distinct photo
TIER_THUMBNAIL_DUPLICATE = 3     # only engine thumbnail, and duplicate of input

TIER_LABELS = {
    TIER_PAGE_IMAGE_UNIQUE: "page image, distinct photo",
    TIER_PAGE_IMAGE_DUPLICATE: "page image, duplicate of input",
    TIER_THUMBNAIL_UNIQUE: "Engine thumbnail only (unverified against source page), distinct photo",
    TIER_THUMBNAIL_DUPLICATE: "Engine thumbnail only (unverified against source page), duplicate of input",
}


def _to_bytes(image_input: str | bytes) -> bytes | None:
    """Best-effort normalize a path-or-bytes input into raw bytes for hashing."""
    if isinstance(image_input, (bytes, bytearray)):
        return bytes(image_input)
    if isinstance(image_input, str) and os.path.exists(image_input):
        try:
            with open(image_input, "rb") as f:
                return f.read()
        except Exception:
            return None
    return None


def _average_hash(image_bytes: bytes | None, hash_size: int = 8) -> int | None:
    """
    Simple perceptual average-hash (aHash). Used only to flag near-duplicate
    images (e.g. the same photo reposted/mirrored elsewhere) - NOT used for
    identity matching, which stays on the face embedding.
    """
    if not image_bytes:
        return None
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((hash_size, hash_size))
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = 0
        for i, p in enumerate(pixels):
            if p > avg:
                bits |= (1 << i)
        return bits
    except Exception:
        return None


def _hamming(a: int | None, b: int | None) -> int | None:
    if a is None or b is None:
        return None
    return bin(a ^ b).count("1")


NEAR_DUPLICATE_HAMMING_THRESHOLD = 6


class MatcherResult:
    """Encapsulates the result of the search and face matching process."""

    def __init__(
        self,
        accepted_record: dict[str, Any] | None,
        candidate_logs: list[dict[str, Any]],
        imgbb_url: str | None = None,
        search_engine: str = "yandex",
        total_engine_matches: int = 0,
        total_social_candidates: int = 0,
        total_web_candidates: int = 0,
        accepted_distance: float | None = None,
        accepted_tier: int | None = None,
        reason: str = "",
    ):
        self.accepted_record = accepted_record
        self.candidate_logs = candidate_logs
        self.imgbb_url = imgbb_url
        self.search_engine = search_engine
        self.total_engine_matches = total_engine_matches
        self.total_lens_matches = total_engine_matches  # Backward compatibility alias
        self.total_social_candidates = total_social_candidates
        self.total_web_candidates = total_web_candidates
        self.accepted_distance = accepted_distance
        self.accepted_tier = accepted_tier
        self.reason = reason

    @property
    def is_match_found(self) -> bool:
        return self.accepted_record is not None


def _evaluate_candidate(
    candidate: dict[str, Any],
    idx: int,
    input_embedding: np.ndarray,
    input_ahash: int | None,
    tol: float,
) -> dict[str, Any]:
    """Fetches a single candidate URL, runs face-check + near-dup check, returns a log dict."""
    url = candidate.get("link", "")
    platform = get_social_platform(url) or "web"
    log: dict[str, Any] = {
        "position": idx,
        "url": url,
        "platform": platform,
        "title": candidate.get("title", ""),
        "thumbnail": candidate.get("thumbnail", ""),
        "author": "",
        "text": "",
        "distance": None,
        "matched": False,
        "verification_source": None,
        "near_duplicate_of_input": None,
        "tier": None,
        "status": "PROCESSING",
    }

    try:
        post_data = fetch_post(url)
        log["author"] = post_data.get("author", "")
        log["text"] = post_data.get("text", "")

        img_bytes = post_data.get("_image_bytes")
        verification_source = "page_image" if img_bytes else None

        # 1. Try downloading direct original page image if provided by search engine (e.g. Yandex)
        if not img_bytes and candidate.get("original_image"):
            try:
                orig_resp = _http_get(candidate["original_image"], timeout=10)
                if orig_resp.status_code == 200 and len(orig_resp.content) > 0:
                    img_bytes = orig_resp.content
                    post_data["_image_bytes"] = img_bytes
                    verification_source = "page_image"
            except Exception:
                pass

        # 2. Fallback to search engine thumbnail ONLY if real page image is unavailable
        if not img_bytes and candidate.get("thumbnail"):
            try:
                thumb_resp = _http_get(candidate["thumbnail"], timeout=10)
                if thumb_resp.status_code == 200 and len(thumb_resp.content) > 0:
                    img_bytes = thumb_resp.content
                    post_data["_image_bytes"] = img_bytes
                    verification_source = "thumbnail_fallback"
            except Exception:
                pass

        if img_bytes:
            log["_image_bytes"] = img_bytes
        log["verification_source"] = verification_source

        if not img_bytes:
            log["status"] = "REJECTED: No image could be retrieved from post or thumbnail."
            return log

        try:
            post_embedding = encode_face(img_bytes)
        except NoFaceFound:
            log["status"] = "REJECTED: No face detected in candidate image."
            return log
        except ImageReadError:
            log["status"] = "REJECTED: Candidate image bytes could not be decoded."
            return log

        dist = cosine_distance(input_embedding, post_embedding)
        log["distance"] = round(dist, 4)

        cand_ahash = _average_hash(img_bytes)
        ham = _hamming(input_ahash, cand_ahash)
        is_dup = ham is not None and ham <= NEAR_DUPLICATE_HAMMING_THRESHOLD
        log["near_duplicate_of_input"] = is_dup
        log["_ahash_hamming"] = ham

        if verification_source == "page_image":
            tier = TIER_PAGE_IMAGE_DUPLICATE if is_dup else TIER_PAGE_IMAGE_UNIQUE
        else:
            tier = TIER_THUMBNAIL_DUPLICATE if is_dup else TIER_THUMBNAIL_UNIQUE
        log["tier"] = tier

        if dist < tol:
            log["matched"] = True
            dup_note = " [near-duplicate of input photo]" if is_dup else " [distinct photo]"
            src_note = " (verified from source page)" if verification_source == "page_image" else " (WARNING: engine thumbnail fallback)"
            log["status"] = f"MATCH (cosine distance {dist:.4f} < {tol}){src_note}{dup_note}"
        else:
            log["status"] = f"REJECTED: cosine distance {dist:.4f} >= {tol} (different person)."

        return log

    except Exception as exc:
        log["status"] = f"ERROR: Failed during extraction/verification: {exc}"
        return log


def _fetch_engine_results(
    engine: str,
    imgbb_url: str,
    serpapi_key: str | None,
) -> tuple[list[dict[str, Any]], str]:
    """Fetches candidates from Yandex, Google Lens, or Hybrid combination."""
    if engine.lower() == "google_lens":
        results = search_google_lens(imgbb_url, api_key=serpapi_key)
        return results, "Google Lens"
    elif engine.lower() == "hybrid":
        # Query Yandex first, supplement with Google Lens
        y_results = []
        try:
            y_results = search_yandex_images(imgbb_url, api_key=serpapi_key)
        except Exception:
            pass

        l_results = []
        try:
            l_results = search_google_lens(imgbb_url, api_key=serpapi_key)
        except Exception:
            pass

        merged = y_results + [r for r in l_results if not any(y["link"] == r["link"] for y in y_results)]
        return merged, f"Hybrid (Yandex {len(y_results)} + Lens {len(l_results)})"
    else:
        # Default: Yandex Images (Gold standard for human face reverse search)
        try:
            results = search_yandex_images(imgbb_url, api_key=serpapi_key)
            if results:
                return results, "Yandex Images"
        except Exception:
            pass

        # Automatic fallback to Google Lens if Yandex raises or returns empty
        results = search_google_lens(imgbb_url, api_key=serpapi_key)
        return results, "Google Lens (Yandex Fallback)"


def find_verified_social_post(
    input_embedding: np.ndarray,
    image_path_or_bytes: str | bytes,
    tol: float = 0.35,
    engine: str = "yandex",
    serpapi_key: str | None = None,
    imgbb_key: str | None = None,
    max_candidates: int = 35,
    until_success: bool = False,
    include_general_web: bool = True,
    offline_demo: bool = False,
    offline_post_url: str | None = None,
) -> MatcherResult:
    """
    Executes Stage 2:
    1. Upload scan (face crop) to ImgBB -> public URL
    2. Query Visual Search Engine (Yandex Images primary / Google Lens / Hybrid)
    3. Rank candidates: social-platform matches first, then general web matches
    4. Evaluate candidate pool:
       - If `until_success=True`: evaluates all 300+ candidates until the first verified face match is found.
       - If `until_success=False`: evaluates up to `max_candidates` and selects the best evidence tier/distance.
    """
    candidate_logs: list[dict[str, Any]] = []

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
            "position": 1, "url": demo_url, "platform": "x", "distance": 0.12,
            "matched": True, "verification_source": "page_image",
            "near_duplicate_of_input": False, "tier": TIER_PAGE_IMAGE_UNIQUE,
            "status": "ACCEPTED (Offline Demo Mode)",
        })
        return MatcherResult(
            accepted_record=post_data, candidate_logs=candidate_logs,
            imgbb_url="https://i.ibb.co/demo/scan.jpg", search_engine="offline_demo",
            total_engine_matches=1, total_social_candidates=1, total_web_candidates=0,
            accepted_distance=0.12, accepted_tier=TIER_PAGE_IMAGE_UNIQUE,
            reason="Matched via offline demo mode.",
        )

    input_bytes = _to_bytes(image_path_or_bytes)
    input_ahash = _average_hash(input_bytes)

    imgbb_url = upload_to_imgbb(image_path_or_bytes, api_key=imgbb_key)
    raw_results, active_engine_label = _fetch_engine_results(engine, imgbb_url, serpapi_key)
    total_engine_matches = len(raw_results)

    if not raw_results:
        return MatcherResult(
            accepted_record=None, candidate_logs=candidate_logs, imgbb_url=imgbb_url,
            search_engine=active_engine_label,
            total_engine_matches=0, total_social_candidates=0, total_web_candidates=0,
            reason=f"No visual matches returned by {active_engine_label} for this image.",
        )

    # Tier 1: social-platform candidates. Tier 2: everything else on the open web.
    social_candidates: list[dict[str, Any]] = []
    web_candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for item in raw_results:
        link = item.get("link", "")
        if not link:
            continue
        canon_link = normalize_url(link)
        if canon_link in seen_urls:
            continue
        seen_urls.add(canon_link)
        if is_social_media_url(canon_link):
            social_candidates.append(item)
        elif include_general_web:
            web_candidates.append(item)

    total_social_candidates = len(social_candidates)
    total_web_candidates = len(web_candidates)

    ordered_candidates = social_candidates + web_candidates
    if not ordered_candidates:
        return MatcherResult(
            accepted_record=None, candidate_logs=candidate_logs, imgbb_url=imgbb_url,
            search_engine=active_engine_label,
            total_engine_matches=total_engine_matches, total_social_candidates=0,
            total_web_candidates=0,
            reason=f"{active_engine_label} returned matches, but none were usable as web/social candidates.",
        )

    # If until_success is True, search the full candidate pool (all 300+ items)
    if until_success:
        candidates_to_eval = ordered_candidates
    else:
        candidates_to_eval = ordered_candidates[:max_candidates]

    passing: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates_to_eval, start=1):
        log = _evaluate_candidate(candidate, idx, input_embedding, input_ahash, tol)
        candidate_logs.append(log)
        if log.get("matched"):
            passing.append(log)
            # In "until_success" mode, stop as soon as we find a confirmed match
            if until_success:
                break

    if not passing:
        return MatcherResult(
            accepted_record=None, candidate_logs=candidate_logs, imgbb_url=imgbb_url,
            search_engine=active_engine_label,
            total_engine_matches=total_engine_matches,
            total_social_candidates=total_social_candidates,
            total_web_candidates=total_web_candidates,
            reason=f"Evaluated {len(candidate_logs)} candidate(s) via {active_engine_label} "
                   f"(until_success={until_success}), none satisfied tolerance (tol={tol}).",
        )

    # Best evidence tier first, then lowest distance within that tier.
    passing.sort(key=lambda l: (l["tier"], l["distance"]))
    best = passing[0]

    # Reconstruct the accepted post record by re-fetching
    accepted_post = fetch_post(best["url"])
    if not accepted_post.get("_image_bytes") and best.get("_image_bytes"):
        accepted_post["_image_bytes"] = best["_image_bytes"]

    return MatcherResult(
        accepted_record=accepted_post,
        candidate_logs=candidate_logs,
        imgbb_url=imgbb_url,
        search_engine=active_engine_label,
        total_engine_matches=total_engine_matches,
        total_social_candidates=total_social_candidates,
        total_web_candidates=total_web_candidates,
        accepted_distance=best["distance"],
        accepted_tier=best["tier"],
        reason=f"Best match via {active_engine_label}: {best['url']} (distance {best['distance']:.4f}, "
               f"evidence tier: {TIER_LABELS[best['tier']]}, evaluated {len(candidate_logs)} candidates).",
    )
