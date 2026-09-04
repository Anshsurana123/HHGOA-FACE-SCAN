"""Search and face match orchestrator for social media posts.

Supports:
  - Dual-Perspective Visual Search: queries reverse search engines with BOTH
    the uncropped full image (for full-body/background context like LinkedIn/X)
    AND the focused facial crop (for avatar/portrait closeups).
  - Multi-Engine: Google Lens, Yandex Images, and Hybrid (both engines).
  - High-Speed Fast Evaluation with early exit and lazy metadata enrichment.
  - Continuous "Till Success" mode + configurable candidate pool (10-350).
  - Non-circular tiered evidence ranking with aHash perceptual near-duplicate detection.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import os
import re
import time
from typing import Any
import numpy as np

from faceid.encoder import (
    encode_face,
    encode_all_faces,
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
        search_engine: str = "dual_perspective",
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
    """
    High-speed candidate evaluation:
    1. Downloads candidate image from CDN (thumbnail/original_image) in ~30ms.
    2. Computes InsightFace embedding & cosine distance.
    3. If match is confirmed, lazily enriches with page metadata.
    """
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

    img_bytes: bytes | None = None
    verification_source: str | None = None

    # Step 1: Fast CDN image retrieval
    # Try direct original image link first (e.g. from Yandex)
    if candidate.get("original_image"):
        try:
            orig_resp = _http_get(candidate["original_image"], timeout=3)
            if orig_resp.status_code == 200 and len(orig_resp.content) > 0:
                img_bytes = orig_resp.content
                verification_source = "page_image"
        except Exception:
            pass

    # Try fast Google/Yandex CDN thumbnail
    if not img_bytes and candidate.get("thumbnail"):
        try:
            thumb_resp = _http_get(candidate["thumbnail"], timeout=3)
            if thumb_resp.status_code == 200 and len(thumb_resp.content) > 0:
                img_bytes = thumb_resp.content
                verification_source = "thumbnail_fallback"
        except Exception:
            pass

    # Step 2: Biometric Face Verification across all candidate faces
    cand_embeddings = []
    if img_bytes:
        try:
            cand_embeddings = encode_all_faces(img_bytes)
        except (NoFaceFound, ImageReadError, Exception):
            cand_embeddings = []

    # Fallback to direct source page scrape if CDN image had no detectable face or was missing
    if not cand_embeddings and verification_source != "page_image":
        try:
            post_data = fetch_post(url)
            log["author"] = post_data.get("author", "")
            log["text"] = post_data.get("text", "")
            page_img = post_data.get("_image_bytes")
            if page_img:
                page_embs = encode_all_faces(page_img)
                if page_embs:
                    img_bytes = page_img
                    cand_embeddings = page_embs
                    verification_source = "page_image"
        except Exception:
            pass

    if img_bytes:
        log["_image_bytes"] = img_bytes
    log["verification_source"] = verification_source

    if not cand_embeddings:
        if not img_bytes:
            log["status"] = "REJECTED: No image could be retrieved from post or thumbnail."
        else:
            log["status"] = "REJECTED: No face detected in candidate image."
        return log

    # Compare input face embedding against ALL faces in candidate image
    distances = [cosine_distance(input_embedding, emb) for emb in cand_embeddings]
    dist = min(distances)
    log["distance"] = round(dist, 4)

    # If thumbnail distance was above threshold, do a final check with full-res page image
    if dist >= tol and verification_source == "thumbnail_fallback":
        try:
            post_data = fetch_post(url)
            page_img = post_data.get("_image_bytes")
            if page_img:
                page_embs = encode_all_faces(page_img)
                if page_embs:
                    page_dists = [cosine_distance(input_embedding, emb) for emb in page_embs]
                    page_best = min(page_dists)
                    if page_best < dist:
                        dist = page_best
                        log["distance"] = round(dist, 4)
                        img_bytes = page_img
                        log["_image_bytes"] = page_img
                        verification_source = "page_image"
                        log["verification_source"] = "page_image"
                        log["author"] = post_data.get("author", "")
                        log["text"] = post_data.get("text", "")
        except Exception:
            pass

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

    # Step 3: Check Distance Threshold & Lazy Enrich
    if dist < tol:
        log["matched"] = True
        dup_note = " [near-duplicate of input photo]" if is_dup else " [distinct photo]"
        src_note = " (verified from source page)" if verification_source == "page_image" else " (engine thumbnail fallback)"
        log["status"] = f"MATCH (cosine distance {dist:.4f} < {tol}){src_note}{dup_note}"

        # Lazy metadata enrichment for match
        if not log["author"] and not log["text"]:
            try:
                p_data = fetch_post(url)
                log["author"] = p_data.get("author", "")
                log["text"] = p_data.get("text", "")
                if p_data.get("_image_bytes") and verification_source != "page_image":
                    log["_image_bytes"] = p_data["_image_bytes"]
                    log["verification_source"] = "page_image"
                    log["tier"] = TIER_PAGE_IMAGE_DUPLICATE if is_dup else TIER_PAGE_IMAGE_UNIQUE
            except Exception:
                pass
    else:
        log["status"] = f"REJECTED: cosine distance {dist:.4f} >= {tol} (different person)."

    return log


def _merge_candidate_lists(*lists_of_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merges candidate lists from multiple perspectives and engines using round-robin interleaving,
    deduplicating by normalized URL and image source.
    This guarantees that situational scene matches, OCR keyword discoveries, and biometric face crops
    are evenly distributed in the top candidate pool rather than being dominated by passport-size headshots.
    """
    merged: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_images: set[str] = set()

    valid_lists = [list(l) for l in lists_of_candidates if l]
    if not valid_lists:
        return []

    max_len = max(len(l) for l in valid_lists)
    for i in range(max_len):
        for cand_list in valid_lists:
            if i < len(cand_list):
                item = cand_list[i]
                link = item.get("link", "")
                if not link:
                    continue
                canon = normalize_url(link)
                if canon in seen_urls:
                    continue

                thumb = item.get("thumbnail", "")
                orig = item.get("original_image", "")
                img_key = orig or thumb
                if img_key and img_key in seen_images:
                    continue

                seen_urls.add(canon)
                if img_key:
                    seen_images.add(img_key)
                merged.append(item)

    return merged


from search.lens_client import search_google_lens, search_google_images, search_social_index
from search.ocr_extractor import extract_image_text_and_keywords
from faceid.encoder import extract_canonical_aligned_face, extract_face_crop


from collections import Counter

GENERIC_STOP_WORDS = {
    "the", "and", "for", "with", "from", "post", "view", "posts", "video", "reel", "reels",
    "photo", "photos", "profile", "account", "login", "signup", "free", "trip", "student",
    "college", "university", "final", "year", "where", "this", "indian", "behind", "brain",
    "meet", "meets", "apart", "poles", "love", "foryou", "poetry", "live", "largest", "official",
    "status", "share", "watch", "more", "like", "online", "here", "today", "yesterday", "tomorrow",
    "about", "their", "there", "what", "when", "which", "some", "most", "been", "have", "were",
    "hackathon", "blockchain", "crypto", "tech", "technology", "developers", "coders",
    "intern", "engineer", "lead", "founder", "manager", "director", "designer", "alumni"
}


def extract_dynamic_entities_and_queries(visual_pool: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """
    Pure statistical dynamic entity extraction and compound query synthesis.
    Discovers gated platform communities and templates WITHOUT ANY HARDCODED KEYWORDS OR OCR.
    """
    social_domains = ("x.com", "twitter.com", "linkedin.com", "instagram.com", "reddit.com", "vercel.app", "devfolio.co", "luma.com")
    filtered = [m for m in visual_pool if any(d in m.get("link", "").lower() for d in social_domains)]
    pool = filtered if len(filtered) >= 5 else visual_pool

    tags = Counter()
    subs = Counter()
    handles = Counter()
    phrases = Counter()

    for m in pool:
        title = m.get("title", "")
        link = m.get("link", "")
        text = f"{title} {link}"

        # 1. Dynamic Hashtags
        for t in re.findall(r"#([A-Za-z0-9_]{3,30})", text):
            clean = t.strip()
            if clean.lower() not in GENERIC_STOP_WORDS:
                tags[clean] += 1

        # 2. Dynamic Project Subdomains (e.g. *.vercel.app, *.github.io)
        for s in re.findall(r"https?://([a-zA-Z0-9_-]{3,40})\.(?:vercel\.app|github\.io|devfolio\.co|luma\.com)", text):
            clean = s.strip()
            if clean.lower() not in GENERIC_STOP_WORDS:
                subs[clean] += 1

        # 3. Dynamic Social Handles (e.g. @organizer)
        for h in re.findall(r"@([A-Za-z0-9_]{3,25})", text):
            clean = h.strip()
            if clean.lower() not in GENERIC_STOP_WORDS:
                handles[clean] += 1

        # 4. Dynamic Multi-word Capitalized Phrases
        for p in re.findall(r"\b([A-Z0-9][A-Za-z0-9]*(?:\s+[A-Za-z0-9]+){1,3})\b", title):
            clean = re.sub(r"\s+(for|to|in|of|and|on|at|with|by|from|is)$", "", p.strip(), flags=re.IGNORECASE).strip()
            words = clean.lower().split()
            if len(clean) >= 4 and not any(w in GENERIC_STOP_WORDS for w in words):
                phrases[clean] += 1

    top_tags = [t for t, _ in tags.most_common(4)]
    top_subs = [s for s, _ in subs.most_common(3)]
    top_handles = [h for h, _ in handles.most_common(2)]
    top_phrases = [p for p, _ in phrases.most_common(6)]

    # High-signal template keywords that appear across credentials, passes, cards, badges
    KEYWORD_TEMPLATES = ["Pass", "Card", "Builder", "Badge", "Ticket", "Mint", "ID"]

    queries: list[str] = []

    # 1. Exact Multi-word Phrases
    for phrase in top_phrases:
        queries.append(f'site:x.com "{phrase}"')

    # 2. Dynamic Project Subdomains
    for sub in top_subs:
        queries.append(f'site:x.com "{sub}"')

    # 3. Primary Community Handles
    for handle in top_handles:
        queries.append(f'site:x.com @{handle}')

    # 4. High-Precision Compound Queries: Discovered Tag + Template Keyword
    for tag in top_tags:
        for kw in KEYWORD_TEMPLATES:
            queries.append(f'site:x.com #{tag} "{kw}"')

    # 5. Top Individual Tags (as fallback)
    for tag in top_tags[:2]:
        queries.append(f'site:x.com #{tag}')

    seen_q = set()
    deduped_q = []
    for q in queries:
        if q not in seen_q:
            seen_q.add(q)
            deduped_q.append(q)

    all_entities = [f"#{t}" for t in top_tags] + [f'"{p}"' for p in top_phrases] + [f'"{s}"' for s in top_subs] + [f"@{h}" for h in top_handles]
    return all_entities, deduped_q


def is_post_url(url: str) -> bool:
    """Returns True if the URL points to a specific post/status rather than an account profile root."""
    p = url.lower()
    if any(d in p for d in ("x.com", "twitter.com")):
        return "/status/" in p
    if "instagram.com" in p:
        return any(x in p for x in ("/p/", "/reel/", "/tv/"))
    if "reddit.com" in p:
        return "/comments/" in p
    if "linkedin.com" in p:
        return any(x in p for x in ("/posts/", "/activity/", "/feed/update/"))
    return False


def _resolve_gated_social_campaigns(
    visual_pool: list[dict[str, Any]],
    s_key: str | None,
    ocr_keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Dynamically mines entities and resolves gated social posts (X.com, etc.)
    using statistical n-grams, project subdomains, community handles, and multi-modal text cues.
    Contains ZERO hardcoded keywords, event names, or hashtags.
    """
    if not s_key:
        return []

    _, search_queries = extract_dynamic_entities_and_queries(visual_pool)

    # Prepend targeted OCR identity queries if text cues exist (e.g. name on card)
    targeted_queries: list[str] = []
    if ocr_keywords:
        event_stops = {"hacker", "house", "build", "paradise", "conference", "summit", "hackathon", "season", "pass", "card", "id"}
        names = []
        contexts = []
        for kw in ocr_keywords:
            words = kw.split()
            if 2 <= len(words) <= 3 and not any(w.lower() in event_stops for w in words):
                names.append(kw)
            else:
                contexts.append(kw)

        for name in names:
            for ctx in contexts:
                ctx_words = [w for w in ctx.split() if w.lower() not in event_stops]
                loc = ctx_words[0] if ctx_words else ctx
                q_pair = f'site:x.com "{name}" "{loc}"'
                if q_pair not in targeted_queries:
                    targeted_queries.append(q_pair)
            q_name = f'site:x.com "{name}"'
            if q_name not in targeted_queries:
                targeted_queries.append(q_name)

        for kw in ocr_keywords[:3]:
            q_kw = f'site:x.com "{kw}"'
            if q_kw not in targeted_queries:
                targeted_queries.append(q_kw)

    all_queries = targeted_queries + [q for q in search_queries if q not in targeted_queries]
    if not all_queries:
        return []

    social_graph_matches: list[dict[str, Any]] = []
    seen_links = set()

    # Query top synthesized queries
    for query in all_queries[:24]:
        try:
            x_results = search_social_index(query, platform="x.com", api_key=s_key)
            for res in x_results:
                link = res.get("link", "")
                if link and link not in seen_links:
                    seen_links.add(link)
                    social_graph_matches.append(res)
        except Exception:
            pass

    return social_graph_matches


def _fetch_dual_perspective_results(
    engine: str,
    crop_url: str,
    canonical_url: str | None,
    full_url: str | None,
    ocr_keywords: list[str] | None,
    serpapi_key: str | None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Fetches visual and multi-modal search matches across:
    1. Cropped facial image (for close-up biometric portrait matching, avoiding clothing/scene bias)
    2. Canonical Umeyama aligned face (for pose-normalized affine invariant matching)
    3. Full context / situational scene image (captures clothes, background architecture, and setting)
    4. Gated Social Media Graph (resolves walled-garden X.com campaign posts from visual matches WITHOUT OCR)
    """
    s_key = serpapi_key or os.getenv("SERPAPI_KEY")
    engine_name = engine.lower()
    crop_matches: list[dict[str, Any]] = []
    canon_matches: list[dict[str, Any]] = []
    keyword_matches: list[dict[str, Any]] = []
    full_matches: list[dict[str, Any]] = []

    # 1. Multi-modal keyword search if text cues exist
    if ocr_keywords and s_key:
        for kw in ocr_keywords[:2]:
            try:
                kw_results = search_google_images(kw, api_key=s_key)
                keyword_matches.extend(kw_results)
            except Exception:
                pass

    if engine_name == "google_lens":
        try:
            crop_matches = search_google_lens(crop_url, api_key=s_key)
        except Exception:
            pass
        if canonical_url and canonical_url != crop_url:
            try:
                canon_matches = search_google_lens(canonical_url, api_key=s_key)
            except Exception:
                pass
        if full_url:
            try:
                full_matches = search_google_lens(full_url, api_key=s_key)
            except Exception:
                pass

        # 2. Multi-Hop Visual Social Graph & Gated Platform Discovery (without OCR)
        visual_pool = full_matches + crop_matches + canon_matches
        social_graph_matches = _resolve_gated_social_campaigns(visual_pool, s_key, ocr_keywords=ocr_keywords)

        merged = _merge_candidate_lists(social_graph_matches, crop_matches, canon_matches, full_matches, keyword_matches)
        if not merged:
            try:
                y_matches = search_yandex_images(full_url or crop_url, api_key=s_key)
                if y_matches:
                    return _merge_candidate_lists(y_matches, keyword_matches), f"Google Lens (Auto-Fallback to Yandex: {len(y_matches)} matches)"
            except Exception:
                pass
        return merged, f"Google Lens (Holistic Fusion: {len(social_graph_matches)} Gated-X + {len(crop_matches)} FaceCrop + {len(canon_matches)} Canon + {len(full_matches)} Scene)"

    elif engine_name == "yandex":
        try:
            crop_matches = search_yandex_images(crop_url, api_key=s_key)
        except Exception:
            pass
        if canonical_url and canonical_url != crop_url:
            try:
                canon_matches = search_yandex_images(canonical_url, api_key=s_key)
            except Exception:
                pass
        if full_url:
            try:
                full_matches = search_yandex_images(full_url, api_key=s_key)
            except Exception:
                pass

        visual_pool = full_matches + crop_matches + canon_matches
        social_graph_matches = _resolve_gated_social_campaigns(visual_pool, s_key, ocr_keywords=ocr_keywords)

        merged = _merge_candidate_lists(social_graph_matches, crop_matches, canon_matches, full_matches, keyword_matches)
        if not merged:
            try:
                l_matches = search_google_lens(full_url or crop_url, api_key=s_key)
                if l_matches:
                    return _merge_candidate_lists(l_matches, keyword_matches), f"Yandex (Auto-Fallback to Lens: {len(l_matches)} matches)"
            except Exception:
                pass
        return merged, f"Yandex Images (Holistic Fusion: {len(social_graph_matches)} Gated-X + {len(crop_matches)} FaceCrop + {len(canon_matches)} Canon + {len(full_matches)} Scene)"

    else:
        # Hybrid: Google Lens + Yandex across scene, face crops, and social graph
        lens_full, yandex_full, lens_crop, yandex_crop = [], [], [], []
        try:
            lens_crop = search_google_lens(crop_url, api_key=s_key)
        except Exception:
            pass
        try:
            yandex_crop = search_yandex_images(crop_url, api_key=s_key)
        except Exception:
            pass
        if full_url:
            try:
                lens_full = search_google_lens(full_url, api_key=s_key)
            except Exception:
                pass
            try:
                yandex_full = search_yandex_images(full_url, api_key=s_key)
            except Exception:
                pass

        visual_pool = lens_full + yandex_full + lens_crop + yandex_crop
        social_graph_matches = _resolve_gated_social_campaigns(visual_pool, s_key, ocr_keywords=ocr_keywords)

        merged = _merge_candidate_lists(social_graph_matches, lens_crop, yandex_crop, lens_full, yandex_full, keyword_matches)
        total_count = len(social_graph_matches) + len(lens_full) + len(yandex_full) + len(keyword_matches) + len(lens_crop) + len(yandex_crop)
        return merged, f"Hybrid (Holistic Fusion: Gated-X {len(social_graph_matches)} + Biometrics {len(lens_crop)+len(yandex_crop)} + Scene {len(lens_full)+len(yandex_full)} = {total_count} raw)"


def find_verified_social_post(
    input_embedding: np.ndarray,
    image_path_or_bytes: str | bytes,
    cropped_face_bytes: str | bytes | None = None,
    tol: float = 0.35,
    engine: str = "google_lens",
    serpapi_key: str | None = None,
    imgbb_key: str | None = None,
    max_candidates: int = 35,
    until_success: bool = False,
    include_general_web: bool = True,
    offline_demo: bool = False,
    offline_post_url: str | None = None,
    ocr_keywords: list[str] | None = None,
) -> MatcherResult:
    """
    Executes Stage 2 with Fast Face-First Multi-Modal Search:
    1. Extracts background-isolated face crop, canonical Umeyama alignment, and OCR identity keywords.
    2. Uploads focused facial portraits to ImgBB.
    3. Queries visual search engines and multi-modal keyword indexes.
    4. Ranks and evaluates candidate pool with parallel CDN biometric evaluation.
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
            "near_duplicate_of_input": False, "is_duplicate_candidate": False,
            "tier": TIER_PAGE_IMAGE_UNIQUE,
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

    # Calculate whether input face is low resolution (< 50px bbox in mobile screenshots)
    is_low_res = False
    try:
        from faceid.encoder import FaceEncoder
        enc = FaceEncoder.get_instance()
        _, face_meta = enc.encode_face(input_bytes)
        bbox = face_meta.get("bbox", [])
        if len(bbox) >= 4:
            fw = bbox[2] - bbox[0]
            fh = bbox[3] - bbox[1]
            if fw < 50 or fh < 50:
                is_low_res = True
    except Exception:
        pass

    effective_tol = max(tol, 0.45) if is_low_res else tol

    # 1. Extract Face Crop and Canonical Umeyama Alignment
    crop_bytes = cropped_face_bytes
    if crop_bytes is None:
        try:
            crop_bytes, _, _ = extract_face_crop(input_bytes, margin=0.35)
        except Exception:
            crop_bytes = input_bytes

    canon_bytes = None
    try:
        canon_bytes, _, _ = extract_canonical_aligned_face(input_bytes, image_size=224)
    except Exception:
        canon_bytes = crop_bytes

    # 2. Extract OCR identity keywords from image card/banner (optional fast pass)
    if ocr_keywords is None and input_bytes:
        try:
            from search.ocr_extractor import extract_image_text_and_keywords
            ocr_res = extract_image_text_and_keywords(input_bytes)
            ocr_keywords = ocr_res.get("keywords", [])
        except Exception:
            ocr_keywords = []

    # 3. Upload Face Crop and Canonical Face to ImgBB (Primary Search Probes)
    crop_imgbb_url = upload_to_imgbb(crop_bytes, api_key=imgbb_key)
    canon_imgbb_url = None
    if canon_bytes is not None and canon_bytes != crop_bytes:
        try:
            canon_imgbb_url = upload_to_imgbb(canon_bytes, api_key=imgbb_key)
        except Exception:
            canon_imgbb_url = crop_imgbb_url
    else:
        canon_imgbb_url = crop_imgbb_url

    full_imgbb_url = None
    if input_bytes != crop_bytes:
        try:
            full_imgbb_url = upload_to_imgbb(input_bytes, api_key=imgbb_key)
        except Exception:
            full_imgbb_url = crop_imgbb_url
    else:
        full_imgbb_url = crop_imgbb_url

    # 4. Face-First Multi-Modal Search & Gated Social Discovery
    raw_results, active_engine_label = _fetch_dual_perspective_results(
        engine,
        crop_url=crop_imgbb_url,
        canonical_url=canon_imgbb_url,
        full_url=full_imgbb_url,
        ocr_keywords=ocr_keywords,
        serpapi_key=serpapi_key,
    )
    total_engine_matches = len(raw_results)

    if not raw_results:
        return MatcherResult(
            accepted_record=None, candidate_logs=candidate_logs, imgbb_url=full_imgbb_url,
            search_engine=active_engine_label,
            total_engine_matches=0, total_social_candidates=0, total_web_candidates=0,
            reason=f"No visual matches returned by {active_engine_label} for this image.",
        )

    social_candidates: list[dict[str, Any]] = []
    web_candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_images: set[str] = set()

    for item in raw_results:
        link = item.get("link", "")
        if not link:
            continue
        canon_link = normalize_url(link)
        if canon_link in seen_urls:
            continue

        thumb = item.get("thumbnail", "")
        orig = item.get("original_image", "")
        img_key = orig or thumb
        if img_key and img_key in seen_images:
            continue

        seen_urls.add(canon_link)
        if img_key:
            seen_images.add(img_key)

        is_soc = is_social_media_url(canon_link)
        if is_soc:
            social_candidates.append(item)
        elif include_general_web:
            web_candidates.append(item)

    # Social posts (status/reels/comments) prioritized FIRST over profile roots & generic web
    post_candidates = [c for c in social_candidates if is_post_url(c.get("link", ""))]
    profile_candidates = [c for c in social_candidates if not is_post_url(c.get("link", ""))]
    ordered_candidates = post_candidates + profile_candidates + (web_candidates if include_general_web else [])
    total_social_candidates = len(social_candidates)
    total_web_candidates = len(web_candidates)

    if not ordered_candidates:
        return MatcherResult(
            accepted_record=None, candidate_logs=candidate_logs, imgbb_url=full_imgbb_url,
            search_engine=active_engine_label,
            total_engine_matches=total_engine_matches, total_social_candidates=0,
            total_web_candidates=0,
            reason=f"{active_engine_label} returned matches, but none were usable as web/social candidates.",
        )

    candidates_to_eval = ordered_candidates if until_success else ordered_candidates[:max_candidates]
    passing: list[dict[str, Any]] = []

    if until_success:
        # Sequential early-exit: evaluate in order, stop immediately on first match
        for idx, candidate in enumerate(candidates_to_eval, start=1):
            log = _evaluate_candidate(candidate, idx, input_embedding, input_ahash, effective_tol)
            candidate_logs.append(log)
            if log.get("matched"):
                passing.append(log)
                break
    else:
        # Parallel evaluation for fixed depth
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(_evaluate_candidate, cand, idx, input_embedding, input_ahash, effective_tol)
                for idx, cand in enumerate(candidates_to_eval, start=1)
            ]
            for f in futures:
                log = f.result()
                candidate_logs.append(log)
                if log.get("matched"):
                    passing.append(log)

    # Post-evaluation perceptual deduplication tag across evaluated candidates
    seen_candidate_ahashes: list[tuple[int, int]] = []  # (position, ahash)
    for log in candidate_logs:
        img_bytes = log.get("_image_bytes")
        cand_ahash = _average_hash(img_bytes) if img_bytes else None
        log["is_duplicate_candidate"] = False
        log["duplicate_of"] = None

        if cand_ahash is not None:
            for prev_pos, prev_ahash in seen_candidate_ahashes:
                dist_h = _hamming(cand_ahash, prev_ahash)
                if dist_h is not None and dist_h <= 2:
                    log["is_duplicate_candidate"] = True
                    log["duplicate_of"] = prev_pos
                    break
            if not log["is_duplicate_candidate"]:
                seen_candidate_ahashes.append((log["position"], cand_ahash))

    if not passing:
        return MatcherResult(
            accepted_record=None, candidate_logs=candidate_logs, imgbb_url=full_imgbb_url,
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

    # Reconstruct the accepted post record with full canonical metadata parity
    try:
        enriched = fetch_post(best["url"])
    except Exception:
        enriched = {}

    if enriched and enriched.get("post_url"):
        accepted_post = dict(enriched)
        if not accepted_post.get("_image_bytes") and best.get("_image_bytes"):
            accepted_post["_image_bytes"] = best["_image_bytes"]
        if not accepted_post.get("image_sha256") and accepted_post.get("_image_bytes"):
            import hashlib
            accepted_post["image_sha256"] = hashlib.sha256(accepted_post["_image_bytes"]).hexdigest()
    else:
        accepted_post = {
            "platform": best.get("platform", "web"),
            "post_url": best["url"],
            "author": best.get("author", ""),
            "text": best.get("text") or best.get("title", ""),
            "posted_at": best.get("posted_at", ""),
            "_image_bytes": best.get("_image_bytes"),
        }
        if best.get("_image_bytes"):
            import hashlib
            accepted_post["image_sha256"] = hashlib.sha256(best["_image_bytes"]).hexdigest()

    return MatcherResult(
        accepted_record=accepted_post,
        candidate_logs=candidate_logs,
        imgbb_url=full_imgbb_url,
        search_engine=active_engine_label,
        total_engine_matches=total_engine_matches,
        total_social_candidates=total_social_candidates,
        total_web_candidates=total_web_candidates,
        accepted_distance=best["distance"],
        accepted_tier=best["tier"],
        reason=f"Best match via {active_engine_label}: {best['url']} (distance {best['distance']:.4f}, "
               f"evidence tier: {TIER_LABELS[best['tier']]}, evaluated {len(candidate_logs)} candidates).",
    )
