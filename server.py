"""FastAPI server for HH-FaceChain Verification Console (Strictly Local)."""

from __future__ import annotations

import os
import io
import json
import time
import shutil
import logging
from typing import Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import numpy as np
from PIL import Image

# Load environment variables
load_dotenv()

# Import existing pipeline modules (no duplication)
from faceid.encoder import (
    encode_face,
    encode_face_with_meta,
    extract_face_crop,
    cosine_distance,
    NoFaceFound,
    ImageReadError,
    FaceIdentificationError,
)
from search.matcher import find_verified_social_post, MatcherResult
from search.post_extractor import fetch_post
from chain.anchor import (
    make_canonical_record,
    compute_record_hash,
    hash_to_bytes32,
)
from chain.local_chain import LocalBlockchain
from chain.web3_client import Web3Client, PolygonAmoyClient

# Initialize directories
BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
OUT_DIR = BASE_DIR / "out"
DEMO_DIR = BASE_DIR / "demo"
DATA_DIR = BASE_DIR / "data"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
UI_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="HH-FaceChain Verification Console",
    description="Forensic Face -> Social Media Search -> Blockchain Verification Pipeline",
    version="2.0.4",
)

# CORS middleware for localhost flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
if OUT_DIR.exists():
    app.mount("/out", StaticFiles(directory=str(OUT_DIR)), name="out")
if DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(DEMO_DIR)), name="demo")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the main Stitch-designed UI."""
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI index.html not found.")
    return FileResponse(str(index_path))


@app.get("/api/sample-images")
async def get_sample_images():
    """Returns available demo images for one-click testing."""
    samples = []
    if DEMO_DIR.exists():
        for img in sorted(DEMO_DIR.glob("*.jpg")):
            name = img.stem
            label = "Barack Obama (Public Scan #1)" if name == "scan1" else "Joe Biden (Public Scan #2)" if name == "scan2" else img.name
            samples.append({
                "id": img.name,
                "label": label,
                "url": f"/demo/{img.name}",
                "filename": img.name,
            })
    return JSONResponse({"samples": samples})


@app.get("/api/chain-status")
async def get_chain_status(network: str = Query("local", enum=["local", "amoy"])):
    """Returns current status and integrity of the selected blockchain."""
    if network == "local":
        chain = LocalBlockchain(filepath=str(DATA_DIR / "local_chain.json"))
        status = chain.get_status()
        return JSONResponse({"network": "local", "status": status})
    else:
        try:
            client = PolygonAmoyClient()
            status = client.get_status()
            return JSONResponse({"network": "amoy", "status": status})
        except Exception as exc:
            return JSONResponse({
                "network": "amoy",
                "status": {
                    "connected": False,
                    "error": str(exc),
                    "status_message": f"Failed to connect to Polygon Amoy: {exc}",
                }
            })


@app.post("/api/run")
async def run_pipeline(
    image: UploadFile = File(None),
    full_image: UploadFile = File(None),
    sample_id: str = Form(None),
    network: str = Form("local"),
    tolerance: float = Form(0.35),
    engine: str = Form("google_lens"),
    max_candidates: int = Form(35),
    until_success: bool = Form(False),
    manual_crop: bool = Form(False),
    offline_demo: bool = Form(False),
):
    """
    Executes the full pipeline:
    1. Face Detection & Embedding Extraction (InsightFace)
    2. Real-time Reverse Search & Re-verification (Yandex / Google Lens / Hybrid via SerpApi + ImgBB)
    3. Blockchain Anchoring (Local PoW or Polygon Amoy)
    """
    logs: list[str] = []
    def log(msg: str):
        timestamp = time.strftime("%H:%M:%S")
        logs.append(f"[{timestamp}] {msg}")

    depth_label = "TILL_SUCCESS (Full Pool)" if until_success else str(max_candidates)
    log(f"INITIATING PIPELINE: network={network.upper()}, engine={engine.upper()}, depth={depth_label}, tolerance={tolerance:.2f}, manual_crop={manual_crop}, offline={offline_demo}")

    # Step 0: Read image bytes
    image_bytes: bytes = b""
    full_search_bytes: bytes = b""
    image_filename = "scan.jpg"

    if image and image.filename:
        image_bytes = await image.read()
        image_filename = image.filename
        log(f"Received uploaded image: {image_filename} ({len(image_bytes)} bytes)")
    elif sample_id:
        sample_path = DEMO_DIR / sample_id
        if sample_path.exists():
            image_bytes = sample_path.read_bytes()
            image_filename = sample_id
            log(f"Loaded demo image: {sample_id} ({len(image_bytes)} bytes)")
        else:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Sample image '{sample_id}' not found.", "logs": logs},
            )
    else:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No image file or sample provided.", "logs": logs},
        )

    # Read original full image if provided alongside manual crop
    if full_image and full_image.filename:
        full_search_bytes = await full_image.read()
        log(f"Preserved full context photo for dual-perspective search ({len(full_search_bytes)} bytes)")
    else:
        full_search_bytes = image_bytes

    # Save scan to out/last_scan.jpg
    last_scan_path = OUT_DIR / "last_scan.jpg"
    with open(last_scan_path, "wb") as f:
        f.write(full_search_bytes if full_search_bytes else image_bytes)

    # =========================================================================
    # STAGE 1: Face Detection & Focused Facial Extraction
    # =========================================================================
    log("--- STAGE 1: Face Identification & Facial Feature Extraction ---")
    start_t = time.time()
    try:
        if manual_crop:
            # User manually framed the face — compute embedding directly on this crop without altering bounds
            cropped_face_bytes = image_bytes
            embedding, face_meta = encode_face_with_meta(image_bytes)
            elapsed_face = time.time() - start_t
            total_faces = face_meta.get("total_faces_detected", 1)
            score = face_meta.get("det_score") or face_meta.get("score") or 0.0
            face_meta["score"] = score
            face_meta["total_faces"] = total_faces
            face_meta["manual_crop"] = True

            # Save exact manual crop to out/cropped_face.jpg
            crop_path = OUT_DIR / "cropped_face.jpg"
            crop_path.write_bytes(cropped_face_bytes)
            cropped_face_url = "/out/cropped_face.jpg"

            log(f"Manual Crop Mode: Preserved user-framed region ({len(cropped_face_bytes)} bytes).")
            log(f"InsightFace confirmed facial embedding in {elapsed_face:.2f}s (confidence: {score:.4f}).")
        else:
            # Automatic face detection and bounding box crop with margin
            cropped_face_bytes, embedding, face_meta = extract_face_crop(image_bytes, margin=0.35)
            elapsed_face = time.time() - start_t
            total_faces = face_meta.get("total_faces_detected", 1)
            score = face_meta.get("det_score") or face_meta.get("score") or 0.0
            face_meta["score"] = score
            face_meta["total_faces"] = total_faces
            
            # Save cropped face to out/cropped_face.jpg
            crop_path = OUT_DIR / "cropped_face.jpg"
            crop_path.write_bytes(cropped_face_bytes)
            cropped_face_url = "/out/cropped_face.jpg"

            log(f"InsightFace detected {total_faces} face(s) in {elapsed_face:.2f}s.")
            log(f"Primary face bbox: {face_meta.get('bbox')}, confidence: {score:.4f}")
            log(f"Extracted focused facial crop ({len(cropped_face_bytes)} bytes) for reverse image search.")
    except NoFaceFound:
        log("ERROR: No human face detected in the input scan.")
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "No human face could be detected in the input image. Please supply a clear portrait or headshot.",
                "logs": logs,
                "stage": 1,
            },
        )
    except Exception as exc:
        log(f"ERROR in Face Detection: {exc}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Face identification failed: {exc}", "logs": logs, "stage": 1},
        )

    # =========================================================================
    # STAGE 2: Reverse Search (Face Portrait) & Re-Verification
    # =========================================================================
    log(f"--- STAGE 2: Facial Reverse Image Search ({engine.upper()}) & Re-Match ---")
    serpapi_key = os.getenv("SERPAPI_KEY")
    imgbb_key = os.getenv("IMGBB_KEY")

    if not offline_demo and not serpapi_key:
        log("WARNING: SERPAPI_KEY is not configured. Falling back to offline mode.")
        offline_demo = True

    start_search_t = time.time()
    try:
        matcher_result: MatcherResult = find_verified_social_post(
            input_embedding=embedding,
            image_path_or_bytes=full_search_bytes,
            cropped_face_bytes=cropped_face_bytes,
            tol=tolerance,
            engine=engine,
            serpapi_key=serpapi_key,
            imgbb_key=imgbb_key,
            max_candidates=max_candidates,
            until_success=until_success,
            offline_demo=offline_demo,
        )
        elapsed_search = time.time() - start_search_t
        log(f"Search executed in {elapsed_search:.2f}s via {matcher_result.search_engine}.")
        log(f"Raw engine matches: {matcher_result.total_engine_matches}, Social candidates: {matcher_result.total_social_candidates}, Web: {matcher_result.total_web_candidates}")

        cand_dir = OUT_DIR / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)

        for idx, cand in enumerate(matcher_result.candidate_logs, start=1):
            cand_url = cand.get("url", "")
            cand_dist = cand.get("distance")
            cand_status = cand.get("status", "")
            dist_str = f"dist={cand_dist:.4f}" if cand_dist is not None else "no_face"
            log(f"Candidate [{cand.get('platform', 'unknown')}]: {cand_url[:50]}... ({dist_str}) -> {cand_status}")

            # Persist candidate image locally if downloaded bytes are present
            if cand.get("_image_bytes"):
                cand_path = cand_dir / f"cand_{idx}.jpg"
                try:
                    cand_path.write_bytes(cand["_image_bytes"])
                    cand["image_url"] = f"/out/candidates/cand_{idx}.jpg"
                except Exception:
                    cand["image_url"] = cand.get("thumbnail", "")
                del cand["_image_bytes"]
            elif cand.get("thumbnail"):
                cand["image_url"] = cand["thumbnail"]
            else:
                cand["image_url"] = None

    except Exception as exc:
        log(f"ERROR in Search/Matcher: {exc}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Search & Match failed: {exc}", "logs": logs, "stage": 2},
        )

    if not matcher_result.is_match_found or not matcher_result.accepted_record:
        log(f"NO MATCH FOUND: {matcher_result.reason}")

        # Find closest candidate to show in preview
        closest_cand = None
        cands_with_dist = [c for c in matcher_result.candidate_logs if c.get("distance") is not None]
        if cands_with_dist:
            closest_cand = min(cands_with_dist, key=lambda c: c["distance"])
        elif matcher_result.candidate_logs:
            closest_cand = matcher_result.candidate_logs[0]

        return JSONResponse(
            content={
                "success": True,
                "matched": False,
                "reason": matcher_result.reason,
                "face": face_meta,
                "input_image_url": "/out/last_scan.jpg",
                "cropped_face_url": cropped_face_url,
                "closest_candidate": closest_cand,
                "candidates": matcher_result.candidate_logs,
                "logs": logs,
                "stage": 2,
            }
        )

    accepted_post = matcher_result.accepted_record
    dist = matcher_result.accepted_distance or 0.0
    accuracy_pct = round(max(0.0, min(100.0, (1.0 - dist) * 100.0)), 1)
    log(f"MATCH ACCEPTED! Platform={accepted_post['platform']}, Distance={dist:.4f} -> Accuracy={accuracy_pct}%")

    # Save matched post image if available
    matched_image_url = None
    if accepted_post.get("_image_bytes"):
        post_img_path = OUT_DIR / "post_image.jpg"
        post_img_path.write_bytes(accepted_post["_image_bytes"])
        matched_image_url = "/out/post_image.jpg"
        log("Saved matched post image to out/post_image.jpg")

    # =========================================================================
    # STAGE 3: Canonical Record & Blockchain Anchoring
    # =========================================================================
    log("--- STAGE 3: Blockchain Canonical Anchoring ---")
    canonical_record = make_canonical_record(accepted_post)
    content_hash = compute_record_hash(canonical_record)
    log(f"Computed Deterministic Content Hash (SHA-256): {content_hash}")

    anchor_info: dict[str, Any] = {
        "network": network,
        "content_hash": content_hash,
        "anchored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if network == "local":
        chain = LocalBlockchain(filepath=str(DATA_DIR / "local_chain.json"))
        try:
            receipt = chain.anchor(
                content_hash=content_hash,
                post_url=accepted_post.get("post_url", "")
            )
        except Exception:
            # If already anchored, retrieve existing proof
            _, proof, _ = chain.verify(content_hash)
            receipt = proof or {
                "block_index": chain.last_block.index,
                "block_hash": chain.last_block.block_hash,
                "prev_hash": chain.last_block.prev_hash,
                "nonce": chain.last_block.nonce,
            }

        anchor_info.update({
            "block_index": receipt.get("block_index"),
            "block_hash": receipt.get("block_hash"),
            "previous_hash": receipt.get("prev_hash"),
            "nonce": receipt.get("nonce"),
            "explorer_url": None,
            "status_label": f"Local PoW Block #{receipt.get('block_index')}",
        })
        log(f"Anchored onto Local PoW Blockchain: Block #{receipt.get('block_index')}, Hash {receipt.get('block_hash')[:16]}...")

    elif network == "amoy":
        try:
            client = PolygonAmoyClient()
            receipt = client.anchor(
                content_hash=content_hash,
                post_url=accepted_post.get("post_url", "")
            )
            tx_hash_val = receipt.get("tx_hash")
            is_already = receipt.get("already_anchored", False)

            if is_already:
                anchor_info.update({
                    "tx_hash": tx_hash_val or "EXISTING_ON_CHAIN_RECORD",
                    "block_number": receipt.get("block_number"),
                    "contract_address": client.contract_address,
                    "explorer_url": receipt.get("explorer_url"),
                    "status_label": "Polygon Amoy (Existing Proof)",
                    "already_anchored": True,
                })
                log(f"Record was already anchored on Polygon Amoy at timestamp {receipt.get('anchored_at_iso')}. Retrieved existing on-chain proof.")
            else:
                anchor_info.update({
                    "tx_hash": tx_hash_val,
                    "block_number": receipt.get("block_number"),
                    "contract_address": client.contract_address,
                    "explorer_url": receipt.get("explorer_url"),
                    "status_label": f"Polygon Amoy Tx {tx_hash_val[:10]}..." if tx_hash_val else "Polygon Amoy",
                    "already_anchored": False,
                })
                log(f"Anchored onto Polygon Amoy: Tx {tx_hash_val}")
        except Exception as exc:
            log(f"ERROR: Polygon Amoy anchoring failed: {exc}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": f"Polygon Amoy anchoring failed: {exc}", "logs": logs, "stage": 3},
            )

    # Save full verified record file
    record_payload = {
        "canonical_record": canonical_record,
        "content_hash": content_hash,
        "face_match": {
            "cosine_distance": dist,
            "accuracy_pct": accuracy_pct,
            "tolerance": tolerance,
        },
        "blockchain_anchor": anchor_info,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    record_file_path = OUT_DIR / "record.json"
    with open(record_file_path, "w", encoding="utf-8") as f:
        json.dump(record_payload, f, indent=2, ensure_ascii=False)
    log("Saved verified record to out/record.json")

    return JSONResponse({
        "success": True,
        "matched": True,
        "face": face_meta,
        "accuracy_pct": accuracy_pct,
        "cosine_distance": dist,
        "post": {
            "platform": accepted_post.get("platform", ""),
            "post_url": accepted_post.get("post_url", ""),
            "author": accepted_post.get("author", ""),
            "text": accepted_post.get("text", ""),
            "posted_at": accepted_post.get("posted_at", ""),
            "image_sha256": accepted_post.get("image_sha256", ""),
        },
        "input_image_url": "/out/last_scan.jpg",
        "cropped_face_url": cropped_face_url,
        "matched_image_url": matched_image_url,
        "anchor": anchor_info,
        "candidates": matcher_result.candidate_logs,
        "logs": logs,
    })


@app.post("/api/verify")
async def verify_record(network: str = Form(None)):
    """
    Performs LIVE re-verification of the saved record:
    1. Reads saved out/record.json
    2. Re-fetches the post live from the web
    3. Recomputes canonical content hash
    4. Verifies hash on selected blockchain
    """
    record_path = OUT_DIR / "record.json"
    if not record_path.exists():
        return JSONResponse(
            status_code=404,
            content={"verified": False, "error": "No record found at out/record.json. Run a scan first."}
        )

    try:
        with open(record_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        saved_canonical = data["canonical_record"]
        saved_hash = data["content_hash"]
        target_network = network or data["blockchain_anchor"].get("network", "local")
        post_url = saved_canonical["post_url"]

        # LIVE re-fetch
        live_post = fetch_post(post_url)
        live_canonical = make_canonical_record(live_post)
        live_hash = compute_record_hash(live_canonical)

        if target_network == "local":
            chain = LocalBlockchain(filepath=str(DATA_DIR / "local_chain.json"))
            verified, proof, msg = chain.verify(live_hash)
            if verified and proof:
                return JSONResponse({
                    "verified": True,
                    "network": "local",
                    "saved_hash": saved_hash,
                    "live_hash": live_hash,
                    "hashes_match": (saved_hash == live_hash),
                    "proof": {
                        "block_index": proof.get("block_index"),
                        "block_hash": proof.get("block_hash"),
                        "anchored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(proof.get("timestamp", 0))),
                        "post_url": proof.get("metadata", {}).get("post_url", post_url),
                    },
                    "message": "Blockchain verification succeeded on Local PoW chain.",
                })
            else:
                return JSONResponse({
                    "verified": False,
                    "network": "local",
                    "saved_hash": saved_hash,
                    "live_hash": live_hash,
                    "hashes_match": (saved_hash == live_hash),
                    "error": f"Live content hash {live_hash} was not found on the local blockchain.",
                })
        else:
            client = PolygonAmoyClient()
            verified, proof, msg = client.verify(live_hash)
            if verified and proof:
                return JSONResponse({
                    "verified": True,
                    "network": "amoy",
                    "saved_hash": saved_hash,
                    "live_hash": live_hash,
                    "hashes_match": (saved_hash == live_hash),
                    "proof": {
                        "contract_address": client.contract_address,
                        "anchored_by": proof.get("by"),
                        "anchored_timestamp": proof.get("anchored_at_iso"),
                        "post_url": proof.get("post_url", post_url),
                        "explorer_url": f"https://amoy.polygonscan.com/address/{client.contract_address}",
                    },
                    "message": "Blockchain verification succeeded on Polygon Amoy testnet.",
                })
            else:
                return JSONResponse({
                    "verified": False,
                    "network": "amoy",
                    "saved_hash": saved_hash,
                    "live_hash": live_hash,
                    "hashes_match": (saved_hash == live_hash),
                    "error": f"Live content hash {live_hash} is not anchored on Polygon Amoy.",
                })

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"verified": False, "error": f"Verification error: {exc}"}
        )


@app.post("/api/tamper")
async def simulate_tamper(network: str = Form(None)):
    """
    Demonstrates tamper-evidence:
    1. Reads saved out/record.json
    2. Mutates post text
    3. Recomputes tampered SHA-256 hash
    4. Shows cryptographic failure when querying the blockchain
    """
    record_path = OUT_DIR / "record.json"
    if not record_path.exists():
        return JSONResponse(
            status_code=404,
            content={"tamper_detected": True, "error": "No record found at out/record.json. Run a scan first."}
        )

    try:
        with open(record_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        saved_canonical = dict(data["canonical_record"])
        original_hash = data["content_hash"]
        target_network = network or data["blockchain_anchor"].get("network", "local")

        original_text = saved_canonical.get("text", "")
        if original_text:
            mutated_text = original_text[:-1] + ("X" if original_text[-1] != "X" else "Y")
        else:
            mutated_text = "TAMPERED_RECORD_CONTENT_FORGED"

        saved_canonical["text"] = mutated_text
        tampered_hash = compute_record_hash(saved_canonical)

        if target_network == "local":
            chain = LocalBlockchain(filepath=str(DATA_DIR / "local_chain.json"))
            verified, proof, msg = chain.verify(tampered_hash)
            return JSONResponse({
                "tamper_detected": True,
                "network": "local",
                "original_text": original_text,
                "mutated_text": mutated_text,
                "original_hash": original_hash,
                "tampered_hash": tampered_hash,
                "on_chain_found": verified,
                "status": "TAMPER DETECTED: Hash mismatch rejected on local blockchain.",
                "reason": f"Tampered content hash '{tampered_hash}' not found in local chain.",
            })
        else:
            client = PolygonAmoyClient()
            verified, proof, msg = client.verify(tampered_hash)
            return JSONResponse({
                "tamper_detected": True,
                "network": "amoy",
                "original_text": original_text,
                "mutated_text": mutated_text,
                "original_hash": original_hash,
                "tampered_hash": tampered_hash,
                "on_chain_found": verified,
                "status": "TAMPER DETECTED: Hash mismatch rejected on Polygon Amoy.",
                "reason": f"Tampered content hash '{tampered_hash}' has not been anchored on-chain.",
            })

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"tamper_detected": True, "error": f"Tamper simulation error: {exc}"}
        )


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("  HH-FaceChain Verification Console (Localhost Presentation UI)")
    print("  Dashboard: http://localhost:8000")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
