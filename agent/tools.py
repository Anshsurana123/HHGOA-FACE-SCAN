"""Agent tools implementation for multi-hop OSINT discovery and biometric verification."""

from __future__ import annotations

import hashlib
import io
import os
import re
import time
from typing import Any, Optional
from urllib.parse import urlparse, urljoin
import numpy as np
import requests
from bs4 import BeautifulSoup

from agent.state import ResearchState, EvidenceNode, normalize_search_query, normalize_target_url
from faceid.encoder import (
    encode_all_faces,
    cosine_distance,
    extract_face_crop,
    DEFAULT_TOLERANCE,
    NoFaceFound,
    ImageReadError,
)
from search.imgbb_client import upload_to_imgbb
from search.lens_client import search_google_lens
from search.yandex_client import search_yandex_images
from search.post_extractor import (
    fetch_post,
    _http_get,
    _extract_opengraph_html,
    normalize_url,
    get_social_platform,
    UA_DESKTOP_CHROME,
    UA_TWITTER_BOT,
)
from search.ocr_extractor import extract_image_text_and_keywords


def _average_hash(image_bytes: bytes, hash_size: int = 8) -> int | None:
    """Computes perceptual aHash for duplicate candidate detection."""
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


def execute_analyze_image(state: ResearchState, args: dict[str, Any]) -> dict[str, Any]:
    """Inspects input image to extract searchable visual clues (scene, clothing, badges)."""
    focus = args.get("focus_area", "full_scene")
    reason = args.get("reason", "")
    state.log(f"[AGENT] Action: analyze_image (focus={focus}) - Reason: {reason}")

    clues = state.visual_clues
    clues["scene_description"] = "Public gathering / conference setting with formal or semi-formal attire."

    # Analyze face metadata if available
    if state.target_face_meta:
        age = state.target_face_meta.get("age")
        gender = "Male" if state.target_face_meta.get("gender") == 1 else "Female"
        clues["demographics"] = f"Apparent {gender}, approx age {age}" if age else gender

    # Run OCR if not already run
    if not state.ocr_results.get("full_text"):
        execute_extract_ocr(state, {"target_region": "full_image", "reason": "Initial text extraction for visual clues"})

    # Extract color / clothing heuristics
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(state.input_image_bytes)).convert("RGB")
        w, h = img.size
        # Sample lower third for clothing color
        lower_crop = img.crop((int(w * 0.2), int(h * 0.7), int(w * 0.8), h))
        stat_colors = lower_crop.resize((1, 1)).getpixel((0, 0))
        r, g, b = stat_colors
        if r > 180 and g > 180 and b > 180:
            color_desc = "light or white clothing"
        elif r < 60 and g < 60 and b < 60:
            color_desc = "dark clothing"
        else:
            color_desc = "colored attire"
        clues["clothing_clues"].append(color_desc)
    except Exception:
        pass

    node = state.evidence_graph.add_node(
        node_type="clue",
        label=f"Visual Analysis ({focus})",
        metadata=dict(clues),
        parent_id=state.root_node.node_id,
        relation="visually_analyzed",
        reason=reason,
    )

    return {
        "status": "success",
        "focus_area": focus,
        "clues": clues,
        "node_id": node.node_id,
    }


def execute_extract_ocr(state: ResearchState, args: dict[str, Any]) -> dict[str, Any]:
    """Runs OCR on the original photo or specific crop regions (lanyard/chest)."""
    region = args.get("target_region", "full_image")
    reason = args.get("reason", "")
    state.log(f"[AGENT] Action: extract_ocr (region={region}) - Reason: {reason}")

    image_to_ocr = state.input_image_bytes

    # If region is chest/lanyard and face bbox is known, crop below face
    if region in ("chest_lanyard", "badge") and state.target_face_meta.get("bbox"):
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(state.input_image_bytes)).convert("RGB")
            w, h = img.size
            bbox = state.target_face_meta["bbox"]
            y2 = int(bbox[3])
            # Crop area below the chin to the bottom
            crop_top = min(h - 10, y2)
            cropped_chest = img.crop((0, crop_top, w, h))
            buf = io.BytesIO()
            cropped_chest.save(buf, format="JPEG")
            image_to_ocr = buf.getvalue()
        except Exception:
            image_to_ocr = state.input_image_bytes

    ocr_res = extract_image_text_and_keywords(image_to_ocr)
    full_text = ocr_res.get("full_text", "").strip()
    keywords = ocr_res.get("keywords", [])

    if full_text:
        state.ocr_results["full_text"] = full_text
    for kw in keywords:
        if kw not in state.ocr_results["keywords"]:
            state.ocr_results["keywords"].append(kw)
            # Register proper noun keywords as discovered entity pivots
            state.add_entity("organization", kw, state.root_node.node_id)

    node = state.evidence_graph.add_node(
        node_type="ocr",
        label=f"OCR ({region}): {keywords[:3]}",
        metadata={"text": full_text, "keywords": keywords, "region": region},
        parent_id=state.root_node.node_id,
        relation="extracted_text",
        reason=reason,
    )

    state.log(f"[OCR] Extracted {len(keywords)} keyword(s): {keywords}")
    return {
        "status": "success",
        "region": region,
        "full_text": full_text,
        "keywords": keywords,
        "node_id": node.node_id,
    }


def execute_reverse_image_search(state: ResearchState, args: dict[str, Any]) -> dict[str, Any]:
    """Executes Google Lens or Yandex reverse image search on face crop or full scene."""
    perspective = args.get("image_perspective", "face_crop")
    engine = args.get("engine", "google_lens")
    reason = args.get("reason", "")
    state.log(f"[AGENT] Action: reverse_image_search ({engine}, perspective={perspective}) - Reason: {reason}")

    # Prepare image bytes
    if perspective == "face_crop":
        try:
            search_bytes, _, _ = extract_face_crop(state.input_image_bytes, margin=0.35)
        except Exception:
            search_bytes = state.input_image_bytes
    else:
        search_bytes = state.input_image_bytes

    imgbb_key = os.getenv("IMGBB_KEY")
    serpapi_key = os.getenv("SERPAPI_KEY")

    if not serpapi_key:
        state.log("[SEARCH] Missing SERPAPI_KEY. Reverse image search skipped.")
        return {"status": "error", "error": "Missing SERPAPI_KEY"}

    try:
        img_url = upload_to_imgbb(search_bytes, api_key=imgbb_key)
        state.log(f"[SEARCH] Uploaded probe to ImgBB: {img_url}")
    except Exception as exc:
        state.log(f"[SEARCH] ImgBB upload failed: {exc}")
        return {"status": "error", "error": f"ImgBB upload failed: {exc}"}

    candidates = []
    try:
        if engine == "yandex":
            candidates = search_yandex_images(img_url, api_key=serpapi_key)
        else:
            candidates = search_google_lens(img_url, api_key=serpapi_key)
    except Exception as exc:
        state.log(f"[SEARCH] Reverse search engine {engine} error: {exc}")
        return {"status": "error", "error": str(exc)}

    # Add results to state search results & collect candidate image URLs
    new_results_count = 0
    new_images_count = 0
    for cand in candidates:
        link = cand.get("link", "")
        if not link:
            continue
        canon = normalize_target_url(link)
        if not any(normalize_target_url(r.get("link", "")) == canon for r in state.search_results):
            state.search_results.append(cand)
            new_results_count += 1

        # Collect thumbnail or original image as candidate image
        img_source = cand.get("original_image") or cand.get("thumbnail")
        if img_source and not state.is_image_seen(img_source):
            state.mark_image_seen(img_source)
            state.candidate_images.append({
                "image_url": img_source,
                "source_page": link,
                "title": cand.get("title", ""),
                "distance": None,
                "matched": False,
            })
            new_images_count += 1

    node = state.evidence_graph.add_node(
        node_type="search",
        label=f"Reverse Search ({engine}, {perspective}): {len(candidates)} matches",
        metadata={"engine": engine, "matches": len(candidates), "image_url": img_url},
        parent_id=state.root_node.node_id,
        relation="reverse_searched",
        reason=reason,
    )

    state.log(f"[SEARCH] Reverse search found {len(candidates)} matches ({new_images_count} new candidate images).")
    return {
        "status": "success",
        "engine": engine,
        "perspective": perspective,
        "total_matches": len(candidates),
        "new_leads": new_results_count,
        "new_candidate_images": new_images_count,
        "node_id": node.node_id,
    }


def execute_web_search(state: ResearchState, args: dict[str, Any]) -> dict[str, Any]:
    """Performs public Google web search via SerpApi with deduplication and state ingestion."""
    query = args.get("query", "").strip()
    domain = args.get("domain_restriction")
    reason = args.get("reason", "")

    if not query:
        return {"status": "error", "error": "Empty search query provided."}

    full_query = f"site:{domain} {query}" if domain and not query.startswith("site:") else query
    state.log(f"[AGENT] Action: web_search (Query: {repr(full_query)}) - Reason: {reason}")

    if state.is_query_seen(full_query):
        state.log(f"[SEARCH] Query already executed in this session. Skipping repetition: {full_query}")
        return {"status": "skipped", "message": "Query already executed previously.", "results": []}

    state.mark_query_seen(full_query)
    serpapi_key = os.getenv("SERPAPI_KEY")

    if not serpapi_key:
        state.log("[SEARCH] Missing SERPAPI_KEY. Web search unavailable.")
        return {"status": "error", "error": "Missing SERPAPI_KEY"}

    endpoint = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": full_query,
        "api_key": serpapi_key,
        "num": 10,
    }
    headers = {"User-Agent": UA_DESKTOP_CHROME}

    results: list[dict[str, Any]] = []
    try:
        resp = requests.get(endpoint, params=params, headers=headers, timeout=state.request_timeout)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("organic_results", []):
                link = item.get("link")
                if link:
                    res_item = {
                        "title": item.get("title", ""),
                        "url": link,
                        "link": link,
                        "snippet": item.get("snippet", ""),
                        "source": "Google Web Search",
                        "query": full_query,
                        "rank": len(results) + 1,
                    }
                    results.append(res_item)
                    if not any(normalize_target_url(r.get("link", "")) == normalize_target_url(link) for r in state.search_results):
                        state.search_results.append(res_item)
    except Exception as exc:
        state.log(f"[SEARCH] Web search request failed: {exc}")
        return {"status": "error", "error": str(exc)}

    node = state.evidence_graph.add_node(
        node_type="query",
        label=f"Search: {full_query[:40]}",
        metadata={"query": full_query, "results_count": len(results)},
        parent_id=state.root_node.node_id,
        relation="queried_web",
        reason=reason,
    )

    state.log(f"[SEARCH] Found {len(results)} search results for query.")
    return {
        "status": "success",
        "query": full_query,
        "results_count": len(results),
        "results": results[:5],
        "node_id": node.node_id,
    }


def execute_search_social_platform(state: ResearchState, args: dict[str, Any]) -> dict[str, Any]:
    """Performs platform-specific social search using site: queries."""
    platform = args.get("platform", "x.com")
    query = args.get("query", "").strip()
    reason = args.get("reason", "")
    full_q = f"site:{platform} {query}" if not query.startswith(f"site:{platform}") else query
    return execute_web_search(state, {"query": full_q, "reason": reason})


def execute_open_url(state: ResearchState, args: dict[str, Any]) -> dict[str, Any]:
    """Inspects a public webpage, parses article text, title, author, and discovered links."""
    url = args.get("url", "").strip()
    reason = args.get("reason", "")
    state.log(f"[AGENT] Action: open_url ({url}) - Reason: {reason}")

    if not url:
        return {"status": "error", "error": "Empty URL supplied."}

    if state.is_url_visited(url):
        state.log(f"[AGENT] URL already visited. Skipping duplicate fetch: {url}")
        return {"status": "skipped", "message": "URL already visited."}

    state.mark_url_visited(url)

    try:
        resp = _http_get(url, timeout=state.request_timeout)
        if resp.status_code != 200:
            state.rejected_urls.add(normalize_target_url(url))
            return {"status": "error", "error": f"HTTP status {resp.status_code}"}

        html = resp.text
        og = _extract_opengraph_html(html)
        soup = BeautifulSoup(html, "html.parser")

        # Strip scripts, styles, and untrusted comments
        for elem in soup(["script", "style", "noscript", "svg"]):
            elem.decompose()

        # Extract readable text summary
        text_content = soup.get_text(separator=" ", strip=True)
        text_summary = re.sub(r"\s+", " ", text_content[:600])

        # Extract potential outgoing links to other social profiles or event pages
        extracted_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("http") and not any(h in href for h in ("google.com", "cookie", "privacy", "terms")):
                extracted_links.append(href)

        # Look for entities in page title or metadata
        title = og.get("text", "") or soup.title.string if soup.title else ""
        platform = get_social_platform(url) or "web"

        node = state.evidence_graph.add_node(
            node_type="url",
            label=f"Page: {urlparse(url).netloc}",
            metadata={"url": url, "title": title[:60], "platform": platform},
            parent_id=state.root_node.node_id,
            relation="inspected_page",
            reason=reason,
        )

        # Automatically harvest images from this page as potential candidate corpus
        execute_extract_page_images(state, {"url": url, "reason": "Automatic image harvesting from inspected page"})

        state.log(f"[AGENT] Successfully inspected {url} (Title: {title[:40]}...)")
        return {
            "status": "success",
            "url": url,
            "title": title,
            "author": og.get("author", ""),
            "text_summary": text_summary,
            "outgoing_links": extracted_links[:10],
            "node_id": node.node_id,
        }

    except Exception as exc:
        state.rejected_urls.add(normalize_target_url(url))
        state.log(f"[AGENT] Failed to open {url}: {exc}")
        return {"status": "error", "error": str(exc)}


def execute_extract_page_images(state: ResearchState, args: dict[str, Any]) -> dict[str, Any]:
    """CRITICAL TOOL: Extracts all candidate photographs from a webpage to populate local image corpus."""
    url = args.get("url", "").strip()
    reason = args.get("reason", "")
    state.log(f"[AGENT] Action: extract_page_images ({url}) - Reason: {reason}")

    if not url:
        return {"status": "error", "error": "Empty URL provided."}

    try:
        resp = _http_get(url, timeout=state.request_timeout)
        if resp.status_code != 200:
            return {"status": "error", "error": f"Failed to fetch page HTML: HTTP {resp.status_code}"}

        soup = BeautifulSoup(resp.text, "html.parser")
        found_urls = set()

        # 1. OpenGraph / Twitter meta image tags
        for meta in soup.find_all("meta"):
            prop = meta.get("property") or meta.get("name") or ""
            if prop in ("og:image", "twitter:image", "twitter:image:src", "image"):
                content = meta.get("content", "").strip()
                if content:
                    found_urls.add(urljoin(url, content))

        # 2. <img> tags (src, data-src, data-original, srcset)
        for img in soup.find_all("img"):
            for attr in ("src", "data-src", "data-original", "data-lazy-src"):
                val = img.get(attr, "").strip()
                if val:
                    found_urls.add(urljoin(url, val))
            # Handle srcset
            srcset = img.get("srcset", "")
            if srcset:
                parts = [p.strip().split()[0] for p in srcset.split(",") if p.strip()]
                if parts:
                    found_urls.add(urljoin(url, parts[-1]))  # pick highest resolution

        # 3. Direct links to images: <a href="*.jpg">
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if any(href.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                found_urls.add(urljoin(url, href))

        # Filter out junk (icons, svg, tracking pixels, 1x1 gifs)
        valid_image_urls = []
        for img_u in found_urls:
            low = img_u.lower()
            if any(j in low for j in ("1x1", "pixel", "icon", "logo", ".svg", "sprite", "avatar_default")):
                continue
            if not img_u.startswith("http"):
                continue
            valid_image_urls.append(img_u)

        # Ingest into candidate image corpus
        new_count = 0
        for img_u in valid_image_urls:
            if not state.is_image_seen(img_u):
                state.mark_image_seen(img_u)
                state.candidate_images.append({
                    "image_url": img_u,
                    "source_page": url,
                    "distance": None,
                    "matched": False,
                })
                new_count += 1

        state.log(f"[AGENT] Extracted {len(valid_image_urls)} image(s) from {url} ({new_count} new to corpus).")
        return {
            "status": "success",
            "source_url": url,
            "total_extracted": len(valid_image_urls),
            "new_added": new_count,
            "sample_images": valid_image_urls[:6],
        }

    except Exception as exc:
        state.log(f"[AGENT] Error extracting images from {url}: {exc}")
        return {"status": "error", "error": str(exc)}


def execute_download_candidate_image(state: ResearchState, args: dict[str, Any]) -> dict[str, Any]:
    """Safely downloads a candidate image and computes perceptual aHash."""
    img_url = args.get("image_url", "").strip()
    source_page = args.get("source_page", "")
    reason = args.get("reason", "")
    state.log(f"[AGENT] Action: download_candidate_image ({img_url[:60]}...) - Reason: {reason}")

    if not img_url:
        return {"status": "error", "error": "Empty image_url."}

    # Check if already downloaded in candidate corpus
    for cand in state.candidate_images:
        if cand.get("image_url") == img_url and cand.get("_bytes"):
            return {"status": "cached", "image_url": img_url, "size_bytes": len(cand["_bytes"])}

    try:
        resp = _http_get(img_url, timeout=state.request_timeout)
        if resp.status_code != 200 or len(resp.content) == 0:
            return {"status": "error", "error": f"Failed to download image (HTTP {resp.status_code})"}

        content = resp.content
        if len(content) > state.max_download_bytes:
            return {"status": "error", "error": f"Image exceeds download limit ({len(content)} > {state.max_download_bytes})"}

        ahash = _average_hash(content)

        # Update candidate record
        matched_cand = None
        for cand in state.candidate_images:
            if cand.get("image_url") == img_url:
                cand["_bytes"] = content
                cand["ahash"] = ahash
                matched_cand = cand
                break

        if not matched_cand:
            matched_cand = {
                "image_url": img_url,
                "source_page": source_page,
                "_bytes": content,
                "ahash": ahash,
                "distance": None,
                "matched": False,
            }
            state.candidate_images.append(matched_cand)

        state.log(f"[AGENT] Downloaded candidate image: {len(content)} bytes (aHash: {ahash})")
        return {
            "status": "success",
            "image_url": img_url,
            "size_bytes": len(content),
            "ahash": ahash,
        }

    except Exception as exc:
        state.log(f"[AGENT] Download image failed: {exc}")
        return {"status": "error", "error": str(exc)}


def execute_face_match(state: ResearchState, args: dict[str, Any]) -> dict[str, Any]:
    """Runs deterministic InsightFace biometric verification against target embedding."""
    img_url = args.get("image_url", "").strip()
    reason = args.get("reason", "")
    state.log(f"[AGENT] Action: face_match on {img_url[:60]}... - Reason: {reason}")

    if not img_url:
        return {"status": "error", "error": "Empty image_url provided."}

    # Find candidate or download
    target_cand = None
    for cand in state.candidate_images:
        if cand.get("image_url") == img_url:
            target_cand = cand
            break

    img_bytes = target_cand.get("_bytes") if target_cand else None
    if not img_bytes:
        dl_res = execute_download_candidate_image(state, {"image_url": img_url, "source_page": "", "reason": "Fetch for face match"})
        if dl_res.get("status") not in ("success", "cached"):
            return {"status": "error", "error": f"Failed to retrieve image: {dl_res.get('error')}"}
        for cand in state.candidate_images:
            if cand.get("image_url") == img_url:
                target_cand = cand
                img_bytes = cand.get("_bytes")
                break

    if not img_bytes:
        return {"status": "error", "error": "Image bytes unavailable for face match."}

    state.total_face_matches += 1

    # InsightFace extraction on all candidate faces
    cand_embeddings = []
    try:
        cand_embeddings = encode_all_faces(img_bytes)
    except (NoFaceFound, ImageReadError, Exception):
        cand_embeddings = []

    if not cand_embeddings:
        if target_cand:
            target_cand["distance"] = 1.0
            target_cand["matched"] = False
        state.log(f"[FACE] No face detected in candidate image {img_url[:50]}")
        return {
            "status": "no_face_detected",
            "matched": False,
            "faces_found": 0,
            "best_distance": 1.0,
        }

    # Biometric comparison
    target_emb = state.target_embedding
    distances = [cosine_distance(target_emb, emb) for emb in cand_embeddings]
    min_dist = float(min(distances))
    best_idx = distances.index(min_dist)
    is_match = min_dist < state.tolerance

    if target_cand:
        target_cand["distance"] = round(min_dist, 4)
        target_cand["matched"] = is_match
        target_cand["faces_found"] = len(cand_embeddings)
        target_cand["matched_face_index"] = best_idx

    source_page = target_cand.get("source_page", "") if target_cand else ""

    node = state.evidence_graph.add_node(
        node_type="match" if is_match else "rejected_face",
        label=f"Face Match: dist={min_dist:.4f} ({'VERIFIED' if is_match else 'DIFFERENT PERSON'})",
        metadata={
            "image_url": img_url,
            "distance": round(min_dist, 4),
            "matched": is_match,
            "source_page": source_page,
            "threshold": state.tolerance,
        },
        parent_id=state.root_node.node_id,
        relation="biometrically_evaluated",
        reason=f"Cosine distance {min_dist:.4f} vs threshold {state.tolerance}",
    )

    if is_match:
        state.log(f"[FACE] *** VERIFIED BIOMETRIC MATCH! *** Distance {min_dist:.4f} < {state.tolerance} (Faces: {len(cand_embeddings)})")
        state.verified_candidates.append(target_cand)
        # Automatically inspect candidate post to resolve full canonical record
        if source_page:
            execute_inspect_candidate_post(state, {
                "post_url": source_page,
                "matched_image_url": img_url,
                "reason": "Resolving verified matching post record",
            })
    else:
        state.log(f"[FACE] Evaluated face: Cosine distance {min_dist:.4f} >= {state.tolerance} (Different person)")

    return {
        "status": "evaluated",
        "matched": is_match,
        "best_distance": round(min_dist, 4),
        "faces_found": len(cand_embeddings),
        "matched_face_index": best_idx,
        "node_id": node.node_id,
    }


def execute_inspect_candidate_post(state: ResearchState, args: dict[str, Any]) -> dict[str, Any]:
    """Resolves matching image into a full canonical post record with metadata & SHA-256."""
    post_url = args.get("post_url", "").strip()
    img_url = args.get("matched_image_url", "").strip()
    reason = args.get("reason", "")
    state.log(f"[AGENT] Action: inspect_candidate_post ({post_url}) - Reason: {reason}")

    if not post_url:
        return {"status": "error", "error": "Empty post_url provided."}

    try:
        enriched = fetch_post(post_url)
    except Exception:
        enriched = {}

    # Locate matched image bytes
    matched_bytes = None
    for cand in state.candidate_images:
        if cand.get("image_url") == img_url and cand.get("_bytes"):
            matched_bytes = cand["_bytes"]
            break

    if not matched_bytes and enriched.get("_image_bytes"):
        matched_bytes = enriched["_image_bytes"]

    img_sha256 = hashlib.sha256(matched_bytes).hexdigest() if matched_bytes else (enriched.get("image_sha256") or "")

    post_record = {
        "platform": enriched.get("platform") or get_social_platform(post_url) or "web",
        "post_url": normalize_url(post_url),
        "author": enriched.get("author", ""),
        "text": enriched.get("text", "") or enriched.get("title", ""),
        "posted_at": enriched.get("posted_at", ""),
        "image_sha256": img_sha256,
        "_image_bytes": matched_bytes,
        "_image_url": img_url,
    }

    state.best_candidate = post_record
    state.log(f"[AGENT] Verified post record compiled: {post_record['post_url']} (Author: {post_record['author']})")

    return {
        "status": "success",
        "post_record": {k: v for k, v in post_record.items() if not k.startswith("_")},
    }


def execute_add_discovered_entity(state: ResearchState, args: dict[str, Any]) -> dict[str, Any]:
    """Registers a newly discovered entity to serve as a search pivot."""
    cat = args.get("category", "organization")
    name = args.get("name", "").strip()
    ctx = args.get("context", "")
    state.add_entity(cat, name)
    return {"status": "success", "category": cat, "name": name}


def execute_finish_investigation(state: ResearchState, args: dict[str, Any]) -> dict[str, Any]:
    """Signals termination of the agent research loop."""
    status = args.get("status", "match_verified")
    summary = args.get("summary", "")
    state.completed = True
    state.termination_reason = summary
    state.log(f"[AGENT] Finish Investigation invoked: {status} - {summary}")
    return {"status": "terminated", "reason": summary}


TOOL_DISPATCH_TABLE = {
    "analyze_image": execute_analyze_image,
    "extract_ocr": execute_extract_ocr,
    "reverse_image_search": execute_reverse_image_search,
    "web_search": execute_web_search,
    "search_social_platform": execute_search_social_platform,
    "open_url": execute_open_url,
    "extract_page_images": execute_extract_page_images,
    "download_candidate_image": execute_download_candidate_image,
    "face_match": execute_face_match,
    "inspect_candidate_post": execute_inspect_candidate_post,
    "add_discovered_entity": execute_add_discovered_entity,
    "finish_investigation": execute_finish_investigation,
}


def dispatch_tool(state: ResearchState, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatches tool execution with strict whitelisting and error containment."""
    func = TOOL_DISPATCH_TABLE.get(tool_name)
    if not func:
        state.log(f"[ERROR] Attempted to call non-whitelisted tool: {tool_name}")
        return {
            "status": "error",
            "error": f"Tool '{tool_name}' is not an allowed action. Allowed tools: {list(TOOL_DISPATCH_TABLE.keys())}",
        }

    try:
        result = func(state, arguments)
        state.action_history.append({
            "tool": tool_name,
            "arguments": arguments,
            "result_status": result.get("status", "unknown"),
            "timestamp": time.time(),
        })
        return result
    except Exception as exc:
        state.log(f"[ERROR] Exception during execution of tool '{tool_name}': {exc}")
        return {"status": "error", "error": f"Tool execution failed: {exc}"}
