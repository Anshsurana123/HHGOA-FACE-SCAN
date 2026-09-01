"""Search and face match orchestrator for social media posts.

v2: fixes two failure modes observed in practice:
  1. "Circular verification" - when a candidate page has no extractable og:image,
     the old code fell back to Google Lens's own thumbnail for the face-check.
     Since Lens picked that thumbnail *because* its whole-image similarity engine
     thought it looked like the input, checking against it just re-confirms
     Lens's own ranking bias rather than independently verifying anything.
     Fix: thumbnail-fallback matches are now tagged and demoted to lowest priority.
  2. "Popularity/near-duplicate bias" - Lens ranks by whole-image similarity and
     page authority, not by face identity, so a popular lookalike (or a mirrored
     repost of the exact input image) can outrank the real, correct post. The old
     code also hard-filtered to 9 social domains and stopped at the first pass
     within only 5-6 evaluated candidates.
     Fix: evaluate the full candidate pool (social + general web, no early return),
     flag near-duplicates of the input via perceptual hashing, and select the best
     (lowest distance) match from the strongest available evidence tier.
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
from search.post_extractor import (
    fetch_post,
    is_social_media_url,
    normalize_url,
    get_social_platform,
    _http_get,
)


# Evidence-strength tiers, best first. Used to rank passing candidates when more
# than one clears the face-match threshold.
TIER_PAGE_IMAGE_UNIQUE = 0       # downloaded from the real post/page, not a dup of the input
TIER_PAGE_IMAGE_DUPLICATE = 1    # downloaded from the real post/page, but same photo as input
TIER_THUMBNAIL_UNIQUE = 2        # only Lens's own thumbnail was available, not a dup of the input
TIER_THUMBNAIL_DUPLICATE = 3     # only Lens's own thumbnail, and it's a dup of the input

TIER_LABELS = {
    TIER_PAGE_IMAGE_UNIQUE: "page image, distinct photo",
    TIER_PAGE_IMAGE_DUPLICATE: "page image, duplicate of input",
    TIER_THUMBNAIL_UNIQUE: "Lens thumbnail only (unverified against source page), distinct photo",
    TIER_THUMBNAIL_DUPLICATE: "Lens thumbnail only (unverified against source page), duplicate of input",
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


# Out of 64 bits (8x8 aHash); <=6 bits differing (~90%+ similar) is treated as
# "same photo, different hosting/compression" rather than a genuinely new image.
NEAR_DUPLICATE_HAMMING_THRESHOLD = 6


class MatcherResult:
    """Encapsulates the result of the search and face matching process."""

    def __init__(
        self,
        accepted_record: dict[str, Any] | None,
        candidate_logs: list[dict[str, Any]],
        imgbb_url: str | None = None,
        total_lens_matches: int = 0,
        total_social_candidates: int = 0,
        total_web_candidates: int = 0,
        accepted_distance: float | None = None,
        accepted_tier: int | None = None,
        reason: str = "",
    ):
        self.accepted_record = accepted_record
        self.candidate_logs = candidate_logs
        self.imgbb_url = imgbb_url
        self.total_lens_matches = total_lens_matches
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

        # Fallback to Lens's own thumbnail ONLY if the real page image is unavailable.
        # This is weaker evidence (see module docstring) and is tagged as such.
        if not img_bytes and candidate.get("thumbnail"):
            try:
                thumb_resp = _http_get(candidate["thumbnail"], timeout=10)
                if thumb_resp.status_code == 200 and len(thumb_resp.content) > 0:
                    img_bytes = thumb_resp.content
                    post_data["_image_bytes"] = img_bytes
                    verification_source = "lens_thumbnail_fallback"
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
            src_note = " (verified from source page)" if verification_source == "page_image" else " (WARNING: only Lens's own thumbnail, not independently confirmed)"
            log["status"] = f"MATCH (cosine distance {dist:.4f} < {tol}){src_note}{dup_note}"
        else:
            log["status"] = f"REJECTED: cosine distance {dist:.4f} >= {tol} (different person)."

        return log

    except Exception as exc:
        log["status"] = f"ERROR: Failed during extraction/verification: {exc}"
        return log


def find_verified_social_post(
    input_embedding: np.ndarray,
    image_path_or_bytes: str | bytes,
    tol: float = 0.35,
    serpapi_key: str | None = None,
    imgbb_key: str | None = None,
    max_candidates: int = 25,
    include_general_web: bool = True,
    offline_demo: bool = False,
    offline_post_url: str | None = None,
) -> MatcherResult:
    """
    Executes Stage 2:
    1. Upload scan (face crop) to ImgBB -> public URL
    2. Query Google Lens via SerpApi
    3. Rank candidates: social-platform matches first, then general web matches
       (previously general-web results were discarded entirely)
    4. Evaluate the FULL candidate pool (up to max_candidates), not just the
       first few in Lens's own rank order - popularity/duplicate-driven ranking
       is not the same as face-identity ranking.
    5. Among all candidates that pass the face-match threshold, select the one
       with the strongest evidence tier (real page image > thumbnail-only;
       distinct photo > duplicate of input), breaking ties by lowest distance.
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
            imgbb_url="https://i.ibb.co/demo/scan.jpg", total_lens_matches=1,
            total_social_candidates=1, total_web_candidates=0,
            accepted_distance=0.12, accepted_tier=TIER_PAGE_IMAGE_UNIQUE,
            reason="Matched via offline demo mode.",
        )

    input_bytes = _to_bytes(image_path_or_bytes)
    input_ahash = _average_hash(input_bytes)

    imgbb_url = upload_to_imgbb(image_path_or_bytes, api_key=imgbb_key)
    lens_results = search_google_lens(imgbb_url, api_key=serpapi_key)
    total_lens_matches = len(lens_results)

    if not lens_results:
        return MatcherResult(
            accepted_record=None, candidate_logs=candidate_logs, imgbb_url=imgbb_url,
            total_lens_matches=0, total_social_candidates=0, total_web_candidates=0,
            reason="No visual matches returned by Google Lens for this image.",
        )

    # Tier 1: social-platform candidates. Tier 2: everything else on the open web.
    social_candidates: list[dict[str, Any]] = []
    web_candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for item in lens_results:
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
            total_lens_matches=total_lens_matches, total_social_candidates=0,
            total_web_candidates=0,
            reason="Google Lens returned matches, but none were usable as web/social candidates.",
        )

    candidates_to_eval = ordered_candidates[:max_candidates]

    passing: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates_to_eval, start=1):
        log = _evaluate_candidate(candidate, idx, input_embedding, input_ahash, tol)
        candidate_logs.append(log)
        if log.get("matched"):
            passing.append(log)

    if not passing:
        return MatcherResult(
            accepted_record=None, candidate_logs=candidate_logs, imgbb_url=imgbb_url,
            total_lens_matches=total_lens_matches,
            total_social_candidates=total_social_candidates,
            total_web_candidates=total_web_candidates,
            reason=f"Evaluated {len(candidates_to_eval)} candidate(s) across social+web, "
                   f"none satisfied the face match tolerance (tol={tol}).",
        )

    # Best evidence tier first, then lowest distance within that tier.
    passing.sort(key=lambda l: (l["tier"], l["distance"]))
    best = passing[0]

    # Reconstruct the accepted post record by re-fetching (cheap, guarantees
    # the record we anchor matches what fetch_post would return at verify time).
    accepted_post = fetch_post(best["url"])
    if not accepted_post.get("_image_bytes") and best.get("_image_bytes"):
        accepted_post["_image_bytes"] = best["_image_bytes"]

    return MatcherResult(
        accepted_record=accepted_post,
        candidate_logs=candidate_logs,
        imgbb_url=imgbb_url,
        total_lens_matches=total_lens_matches,
        total_social_candidates=total_social_candidates,
        total_web_candidates=total_web_candidates,
        accepted_distance=best["distance"],
        accepted_tier=best["tier"],
        reason=f"Best match: {best['url']} (distance {best['distance']:.4f}, "
               f"evidence tier: {TIER_LABELS[best['tier']]}).",
    )
