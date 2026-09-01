"""HH Goa 2026 Task 3 — Face -> Social Media Search -> Blockchain Pipeline CLI."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any
import click
from dotenv import load_dotenv

# Load .env configuration
load_dotenv()
load_dotenv(".env.txt")

from faceid.encoder import (
    FaceEncoder,
    encode_face_with_meta,
    extract_face_crop,
    cosine_distance,
    NoFaceFound,
    ImageReadError,
    DEFAULT_TOLERANCE,
)
from search.matcher import find_verified_social_post, MatcherResult
from search.post_extractor import fetch_post
from chain.anchor import make_canonical_record, compute_record_hash
from chain.local_chain import LocalBlockchain, LocalChainError
from chain.web3_client import Web3Client, Web3ChainError


def _print_banner():
    click.secho("\n" + "=" * 70, fg="cyan", bold=True)
    click.secho("  HH Goa 2026: Face -> Social Search -> Blockchain Pipeline", fg="cyan", bold=True)
    click.secho("=" * 70 + "\n", fg="cyan", bold=True)


def _format_dict(d: dict[str, Any], indent: int = 2) -> str:
    lines = []
    pad = " " * indent
    for k, v in d.items():
        if k.startswith("_"):
            continue
        lines.append(f"{pad}{click.style(k, bold=True)}: {v}")
    return "\n".join(lines)


@click.group()
def cli():
    """Face identification, genuine social media search, and blockchain anchoring pipeline."""
    pass


@cli.command("run")
@click.option("--image", "-i", required=True, type=click.Path(exists=True), help="Path to input face scan image.")
@click.option("--network", "-n", type=click.Choice(["local", "amoy"], case_sensitive=False), default="local", show_default=True, help="Blockchain network to anchor record.")
@click.option("--tol", "-t", type=float, default=DEFAULT_TOLERANCE, show_default=True, help="Cosine distance tolerance threshold for face match.")
@click.option("--engine", "-e", type=click.Choice(["google_lens", "yandex", "hybrid"], case_sensitive=False), default="google_lens", show_default=True, help="Visual reverse search engine (Google Lens is social primary).")
@click.option("--max-candidates", "-m", type=int, default=35, show_default=True, help="Search depth (number of visual candidates to evaluate).")
@click.option("--until-success", is_flag=True, default=False, help="Search continuously across the full 300+ candidate pool until a match is found.")
@click.option("--offline-demo", is_flag=True, default=False, help="Run search in offline demonstration mode without external API calls.")
@click.option("--out-dir", "-o", default="out", show_default=True, help="Directory to save verified record and post image.")
def run_cmd(image: str, network: str, tol: float, engine: str, max_candidates: int, until_success: bool, offline_demo: bool, out_dir: str):
    """Run the complete pipeline end-to-end: Detect -> Search -> Match -> Anchor -> Save."""
    _print_banner()
    click.secho(f"[*] Starting Pipeline Execution", fg="yellow", bold=True)
    click.secho(f"    - Input image: {image}")
    click.secho(f"    - Target network: {network.upper()}")
    click.secho(f"    - Search Engine: {engine.upper()}")
    depth_str = "TILL_SUCCESS (Full Pool)" if until_success else f"{max_candidates} candidates"
    click.secho(f"    - Search Depth: {depth_str}")
    click.secho(f"    - Cosine distance tolerance: {tol}")
    click.secho(f"    - Offline demo mode: {offline_demo}\n")

    # =========================================================================
    # STAGE 1: Face Identification & Embedding Extraction
    # =========================================================================
    click.secho("--- STAGE 1: Face Identification & Facial Crop Extraction ---", fg="blue", bold=True)
    try:
        start_t = time.time()
        cropped_face_bytes, embedding, meta = extract_face_crop(image, margin=0.35)
        elapsed = time.time() - start_t
        bbox_str = ", ".join(f"{int(c)}" for c in meta["bbox"])
        det_score_str = f"{meta['det_score']:.4f}" if meta['det_score'] is not None else "N/A"

        click.secho(f" [OK] Face detected in {elapsed:.2f}s", fg="green")
        click.echo(f"      - Total faces detected: {meta['total_faces_detected']}")
        click.echo(f"      - Selected largest face bbox: [{bbox_str}]")
        click.echo(f"      - Detection confidence score: {det_score_str}")
        click.echo(f"      - Embedding dimensions: {len(embedding)} (L2-normalized)")
        click.echo(f"      - Extracted focused facial crop ({len(cropped_face_bytes)} bytes) for reverse search")
    except NoFaceFound as e:
        click.secho(f" [ERROR] Stage 1 Failed: {e}", fg="red", bold=True)
        sys.exit(1)
    except ImageReadError as e:
        click.secho(f" [ERROR] Stage 1 Failed: {e}", fg="red", bold=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f" [ERROR] Unexpected error in Stage 1: {e}", fg="red", bold=True)
        sys.exit(1)

    # =========================================================================
    # STAGE 2: Genuine Web/Social Reverse Search & Verification
    # =========================================================================
    click.secho(f"\n--- STAGE 2: Genuine Social Media Search ({engine.upper()}) & Face Re-Match ---", fg="blue", bold=True)
    try:
        match_result = find_verified_social_post(
            input_embedding=embedding,
            image_path_or_bytes=image,
            cropped_face_bytes=cropped_face_bytes,
            tol=tol,
            engine=engine,
            max_candidates=max_candidates,
            until_success=until_success,
            offline_demo=offline_demo,
        )

        if match_result.imgbb_url:
            click.echo(f" [*] Uploaded scan to ImgBB: {match_result.imgbb_url}")
        click.echo(f" [*] Search engine used: {match_result.search_engine}")
        click.echo(f" [*] Raw matches found: {match_result.total_engine_matches}")
        click.echo(f" [*] Filtered social candidates: {match_result.total_social_candidates}")
        click.echo(f" [*] Filtered web candidates: {match_result.total_web_candidates}")

        click.secho("\n Candidate Evaluation Logs:", bold=True)
        for cand in match_result.candidate_logs:
            pos = cand.get("position", "-")
            url = cand.get("url", "")
            status = cand.get("status", "")
            dist_info = f" [dist: {cand['distance']}]" if cand.get("distance") is not None else ""
            color = "green" if cand.get("matched") else "yellow"
            click.secho(f"   [{pos}] {url}{dist_info} -> {status}", fg=color)

        if not match_result.is_match_found:
            click.secho(f"\n [INFO] Pipeline completed: {match_result.reason}", fg="yellow", bold=True)
            click.echo(" (No verified face match found is a legitimate outcome.)")
            sys.exit(0)

        post_record = match_result.accepted_record
        dist = match_result.accepted_distance
        click.secho(f"\n [OK] Verified face match confirmed!", fg="green", bold=True)
        click.echo(f"      - Matched Platform: {post_record.get('platform')}")
        click.echo(f"      - Post URL: {post_record.get('post_url')}")
        click.echo(f"      - Author: {post_record.get('author') or 'N/A'}")
        click.echo(f"      - Cosine Distance: {dist:.4f} (threshold: < {tol})")
        click.echo(f"      - Image SHA-256: {post_record.get('image_sha256')}")

    except Exception as e:
        click.secho(f" [ERROR] Stage 2 Failed: {e}", fg="red", bold=True)
        sys.exit(1)

    # =========================================================================
    # STAGE 3: Blockchain Canonical Hashing & Anchoring
    # =========================================================================
    click.secho("\n--- STAGE 3: Blockchain Anchoring ---", fg="blue", bold=True)
    canonical_record = make_canonical_record(post_record)
    content_hash = compute_record_hash(canonical_record)
    post_url = canonical_record["post_url"]

    click.echo(f" [*] Canonical Record:")
    click.echo(_format_dict(canonical_record, indent=6))
    click.secho(f" [*] Deterministic Content Hash (SHA-256): {content_hash}", fg="magenta", bold=True)

    anchor_info: dict[str, Any] = {}
    if network.lower() == "local":
        try:
            chain = LocalBlockchain()
            anchor_info = chain.anchor(content_hash=content_hash, post_url=post_url)
            click.secho(f"\n [OK] Anchored on Local Simulated Blockchain!", fg="green", bold=True)
            click.echo(f"      - Block Index: #{anchor_info['block_index']}")
            click.echo(f"      - Block Hash: {anchor_info['block_hash']}")
            click.echo(f"      - Previous Hash: {anchor_info['prev_hash']}")
            click.echo(f"      - Proof-of-Work Nonce: {anchor_info['nonce']}")
            click.echo(f"      - Timestamp: {anchor_info['anchored_at']}")
        except LocalChainError as e:
            click.secho(f" [WARNING] Local Chain: {e}", fg="yellow")
            anchor_info = {"network": "local", "content_hash": content_hash, "post_url": post_url, "already_anchored": True}
        except Exception as e:
            click.secho(f" [ERROR] Local blockchain anchoring failed: {e}", fg="red", bold=True)
            sys.exit(1)

    elif network.lower() == "amoy":
        try:
            client = Web3Client()
            click.echo(f" [*] Submitting transaction to Polygon Amoy ({client.contract_address})...")
            anchor_info = client.anchor(content_hash=content_hash, post_url=post_url)
            click.secho(f"\n [OK] Anchored on Polygon Amoy Testnet!", fg="green", bold=True)
            click.echo(f"      - Contract: {anchor_info['contract_address']}")
            click.echo(f"      - Transaction Hash: {anchor_info['tx_hash']}")
            click.echo(f"      - Block Number: {anchor_info['block_number']}")
            click.echo(f"      - Explorer Link: {anchor_info['explorer_url']}")
        except Web3ChainError as e:
            click.secho(f" [ERROR] Polygon Amoy anchoring failed: {e}", fg="red", bold=True)
            sys.exit(1)
        except Exception as e:
            click.secho(f" [ERROR] Web3 unexpected error: {e}", fg="red", bold=True)
            sys.exit(1)

    # =========================================================================
    # PERSISTENCE: Save Record and Image to out/
    # =========================================================================
    os.makedirs(out_dir, exist_ok=True)
    record_file = os.path.join(out_dir, "record.json")
    image_file = os.path.join(out_dir, "post_image.jpg")

    saved_data = {
        "canonical_record": canonical_record,
        "content_hash": content_hash,
        "network": network.lower(),
        "offline_demo": offline_demo,
        "anchor_info": anchor_info,
        "face_match": {
            "cosine_distance": match_result.accepted_distance,
            "tolerance": tol,
            "input_scan": image,
        },
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(saved_data, f, indent=2, ensure_ascii=False)

    if post_record.get("_image_bytes"):
        with open(image_file, "wb") as f:
            f.write(post_record["_image_bytes"])

    click.secho(f"\n=== Execution Complete ===", fg="cyan", bold=True)
    click.secho(f" [*] Saved verified record to: {record_file}", fg="green")
    if post_record.get("_image_bytes"):
        click.secho(f" [*] Saved post image to: {image_file}", fg="green")
    click.echo(f"\nRun live verification anytime with:")
    click.secho(f"  python main.py verify --record {record_file} --network {network.lower()}", fg="yellow")
    click.echo(f"Run tamper-evidence demo with:")
    click.secho(f"  python main.py tamper --record {record_file} --network {network.lower()}\n", fg="yellow")


@cli.command("search")
@click.option("--image", "-i", required=True, type=click.Path(exists=True), help="Path to input face scan image.")
@click.option("--tol", "-t", type=float, default=DEFAULT_TOLERANCE, show_default=True, help="Cosine distance tolerance threshold.")
@click.option("--engine", "-e", type=click.Choice(["google_lens", "yandex", "hybrid"], case_sensitive=False), default="google_lens", show_default=True, help="Visual reverse search engine (Google Lens primary).")
@click.option("--max-candidates", "-m", type=int, default=35, show_default=True, help="Search depth (number of visual candidates to evaluate).")
@click.option("--until-success", is_flag=True, default=False, help="Search continuously across the full 300+ candidate pool until a match is found.")
@click.option("--offline-demo", is_flag=True, default=False, help="Run search in offline demonstration mode.")
def search_cmd(image: str, tol: float, engine: str, max_candidates: int, until_success: bool, offline_demo: bool):
    """Execute Stages 1 & 2 only: Detect face, search social platforms, and print match evaluation."""
    _print_banner()
    click.secho(f"[*] Running Face Search Mode (Stages 1 & 2)", fg="yellow", bold=True)
    click.echo(f"    - Input image: {image}")
    click.echo(f"    - Search engine: {engine.upper()}")
    depth_str = "TILL_SUCCESS (Full Pool)" if until_success else f"{max_candidates} candidates"
    click.echo(f"    - Search depth: {depth_str}")
    click.echo(f"    - Tolerance: {tol}\n")

    try:
        cropped_face_bytes, embedding, meta = extract_face_crop(image, margin=0.35)
        click.secho(f" [OK] Face detected: bbox={meta['bbox']}, det_score={meta['det_score']:.4f}", fg="green")
    except Exception as e:
        click.secho(f" [ERROR] Face detection failed: {e}", fg="red", bold=True)
        sys.exit(1)

    try:
        match_result = find_verified_social_post(
            input_embedding=embedding,
            image_path_or_bytes=image,
            cropped_face_bytes=cropped_face_bytes,
            tol=tol,
            engine=engine,
            max_candidates=max_candidates,
            until_success=until_success,
            offline_demo=offline_demo,
        )

        click.echo(f" [*] Search engine used: {match_result.search_engine}")
        click.echo(f" [*] Raw matches found: {match_result.total_engine_matches}")
        click.echo(f" [*] Filtered social candidates: {match_result.total_social_candidates}")
        click.echo(f" [*] Filtered web candidates: {match_result.total_web_candidates}")

        click.secho("\n Candidate Evaluation:", bold=True)
        for cand in match_result.candidate_logs:
            pos = cand.get("position", "-")
            url = cand.get("url", "")
            status = cand.get("status", "")
            dist_info = f" [dist: {cand['distance']}]" if cand.get("distance") is not None else ""
            color = "green" if cand.get("matched") else "yellow"
            click.secho(f"   [{pos}] {url}{dist_info} -> {status}", fg=color)

        if match_result.is_match_found:
            post = match_result.accepted_record
            click.secho(f"\n [ACCEPTED MATCH]", fg="green", bold=True)
            click.echo(f"   - Platform: {post.get('platform')}")
            click.echo(f"   - Post URL: {post.get('post_url')}")
            click.echo(f"   - Author: {post.get('author')}")
            click.echo(f"   - Distance: {match_result.accepted_distance:.4f}")
            click.echo(f"   - Image SHA256: {post.get('image_sha256')}")
        else:
            click.secho(f"\n [NO MATCH FOUND]: {match_result.reason}", fg="yellow", bold=True)

    except Exception as e:
        click.secho(f" [ERROR] Search failed: {e}", fg="red", bold=True)
        sys.exit(1)


@cli.command("verify")
@click.option("--record", "-r", required=True, type=click.Path(exists=True), help="Path to saved record.json.")
@click.option("--network", "-n", type=click.Choice(["local", "amoy"], case_sensitive=False), default=None, help="Blockchain network (defaults to network recorded in JSON).")
@click.option("--offline-demo", is_flag=True, default=False, help="Verify in offline mode using saved canonical record.")
def verify_cmd(record: str, network: str | None, offline_demo: bool):
    """LIVE re-fetch post, recompute canonical hash, and verify against on-chain record."""
    _print_banner()
    click.secho("=== LIVE Blockchain Re-Verification ===", fg="yellow", bold=True)

    with open(record, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_network = (network or data.get("network") or "local").lower()
    is_offline = offline_demo or data.get("offline_demo", False)
    cached_canonical = data.get("canonical_record", {})
    stored_hash = data.get("content_hash", "")
    post_url = cached_canonical.get("post_url", "")

    click.echo(f" [*] Target Network: {target_network.upper()}")
    click.echo(f" [*] Post URL: {post_url}")
    click.echo(f" [*] Saved Content Hash: {stored_hash}\n")

    if not post_url:
        click.secho(" [ERROR] Record file does not contain a valid post_url.", fg="red", bold=True)
        sys.exit(1)

    # Step 1: LIVE re-fetch post using shared function (or offline mode if specified)
    if is_offline:
        click.secho(" [*] Offline Mode: using saved canonical record hash...", fg="blue")
        live_canonical = cached_canonical
        live_hash = compute_record_hash(live_canonical)
    else:
        click.secho(" [*] LIVE re-fetching post metadata from web...", fg="blue")
        try:
            live_post = fetch_post(post_url)
            live_canonical = make_canonical_record(live_post)
            live_hash = compute_record_hash(live_canonical)
            click.echo(f" [*] Recomputed Live Content Hash: {live_hash}")
        except Exception as e:
            click.secho(f" [WARNING] Live fetch failed ({e}). Falling back to cached canonical data.", fg="yellow")
            live_canonical = cached_canonical
            live_hash = compute_record_hash(live_canonical)

    # Compare cached vs live hash
    if live_hash != stored_hash:
        click.secho("\n [FAIL] Live post content has changed since anchoring!", fg="red", bold=True)
        click.echo(f"   Stored Hash: {stored_hash}")
        click.echo(f"   Live Hash:   {live_hash}")
        sys.exit(1)

    # Step 2: Verify on blockchain
    click.secho(f" [*] Verifying hash against {target_network.upper()} blockchain...", fg="blue")
    if target_network == "local":
        chain = LocalBlockchain()
        is_valid, anchor_data, msg = chain.verify(content_hash=live_hash, expected_post_url=post_url)
        if is_valid:
            click.secho(f"\n [PASS] Blockchain Verification Successful!", fg="green", bold=True)
            click.echo(f"   - Network: LOCAL SIMULATED BLOCKCHAIN")
            click.echo(f"   - Block Index: #{anchor_data['block_index']}")
            click.echo(f"   - Block Hash: {anchor_data['block_hash']}")
            click.echo(f"   - Anchored At: {anchor_data['anchored_at_iso']}")
            click.echo(f"   - Post URL: {anchor_data['post_url']}")
            click.echo(f"   - Content Hash: {anchor_data['content_hash']}")
        else:
            click.secho(f"\n [FAIL] Verification Failed: {msg}", fg="red", bold=True)
            sys.exit(1)

    elif target_network == "amoy":
        try:
            client = Web3Client()
            is_valid, anchor_data, msg = client.verify_on_chain(content_hash=live_hash, expected_post_url=post_url)
            if is_valid:
                click.secho(f"\n [PASS] Blockchain Verification Successful!", fg="green", bold=True)
                click.echo(f"   - Network: POLYGON AMOY TESTNET")
                click.echo(f"   - Contract Address: {anchor_data['contract_address']}")
                click.echo(f"   - Anchored By: {anchor_data['by']}")
                click.echo(f"   - Anchored Timestamp: {anchor_data['anchored_at_iso']}")
                click.echo(f"   - Post URL: {anchor_data['post_url']}")
                click.echo(f"   - Content Hash: {anchor_data['content_hash']}")
                click.echo(f"   - Explorer Link: https://amoy.polygonscan.com/address/{anchor_data['contract_address']}")
            else:
                click.secho(f"\n [FAIL] Verification Failed: {msg}", fg="red", bold=True)
                sys.exit(1)
        except Exception as e:
            click.secho(f"\n [FAIL] Web3 verification error: {e}", fg="red", bold=True)
            sys.exit(1)


@cli.command("tamper")
@click.option("--record", "-r", required=True, type=click.Path(exists=True), help="Path to saved record.json.")
@click.option("--network", "-n", type=click.Choice(["local", "amoy"], case_sensitive=False), default=None, help="Blockchain network to verify against.")
def tamper_cmd(record: str, network: str | None):
    """Tamper-evidence demo: mutates cached post text, recomputes hash, and proves on-chain mismatch."""
    _print_banner()
    click.secho("=== Tamper-Evidence Demonstration ===", fg="magenta", bold=True)

    with open(record, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_network = (network or data.get("network") or "local").lower()
    orig_canonical = dict(data.get("canonical_record", {}))
    orig_hash = data.get("content_hash", "")
    post_url = orig_canonical.get("post_url", "")

    click.echo(f" [*] Original Post Text: {repr(orig_canonical.get('text', ''))}")
    click.echo(f" [*] Original Canonical Hash: {orig_hash}\n")

    # Mutate 1 character in text
    orig_text = orig_canonical.get("text", "")
    tampered_text = (orig_text + " [TAMPERED]") if not orig_text else (orig_text[:-1] + "X" if orig_text[-1] != "X" else orig_text[:-1] + "Y")
    
    tampered_canonical = dict(orig_canonical)
    tampered_canonical["text"] = tampered_text
    tampered_hash = compute_record_hash(tampered_canonical)

    click.secho(" [!] Mutating record text to simulate tampering:", fg="yellow", bold=True)
    click.echo(f"     Before: {repr(orig_text)}")
    click.echo(f"     After:  {repr(tampered_text)}\n")

    click.secho(" [*] Side-by-Side Hash Comparison:", bold=True)
    click.echo(f"     Original Hash (On-Chain): {click.style(orig_hash, fg='green')}")
    click.echo(f"     Tampered Hash (Mutated):  {click.style(tampered_hash, fg='red')}\n")

    # Query blockchain for tampered hash
    click.secho(f" [*] Querying {target_network.upper()} blockchain for the tampered hash...", fg="blue")
    if target_network == "local":
        chain = LocalBlockchain()
        is_valid, anchor_data, msg = chain.verify(content_hash=tampered_hash, expected_post_url=post_url)
        if not is_valid:
            click.secho(f"\n [FAIL] TAMPERING DETECTED! Content hash mismatch on-chain.", fg="red", bold=True)
            click.echo(f"   Reason: {msg}")
            click.secho(" [DEMO SUCCESS] The blockchain rejected the altered record with cryptographic proof.\n", fg="green", bold=True)
        else:
            click.secho(" [UNEXPECTED] Tampered hash was found on chain.", fg="red")

    elif target_network == "amoy":
        try:
            client = Web3Client()
            is_valid, anchor_data, msg = client.verify_on_chain(content_hash=tampered_hash, expected_post_url=post_url)
            if not is_valid:
                click.secho(f"\n [FAIL] TAMPERING DETECTED! Content hash mismatch on-chain.", fg="red", bold=True)
                click.echo(f"   Reason: {msg}")
                click.secho(" [DEMO SUCCESS] The blockchain rejected the altered record with cryptographic proof.\n", fg="green", bold=True)
            else:
                click.secho(" [UNEXPECTED] Tampered hash was found on chain.", fg="red")
        except Exception as e:
            click.secho(f" [ERROR] Web3 query failed: {e}", fg="red", bold=True)


@cli.command("chain-status")
@click.option("--network", "-n", type=click.Choice(["local", "amoy"], case_sensitive=False), default="local", show_default=True, help="Blockchain network to inspect.")
def chain_status_cmd(network: str):
    """Show blockchain status and latest anchored blocks/records."""
    _print_banner()
    click.secho(f"=== Blockchain Status: {network.upper()} ===", fg="yellow", bold=True)

    if network.lower() == "local":
        chain = LocalBlockchain()
        status = chain.get_status()
        click.echo(_format_dict(status, indent=2))
        click.secho("\n Recent Blocks:", bold=True)
        for b in chain.chain[-5:]:
            num_records = len(b.records)
            click.echo(f"   Block #{b.index}: hash={b.block_hash[:16]}... nonce={b.nonce} records={num_records}")

    elif network.lower() == "amoy":
        try:
            client = Web3Client()
            click.echo(f"  RPC Endpoint: {client.active_rpc}")
            click.echo(f"  Chain ID: {client.w3.eth.chain_id}")
            click.echo(f"  Contract Address: {client.contract_address}")
            click.echo(f"  Latest Block: {client.w3.eth.block_number}")
            click.echo(f"  Explorer: https://amoy.polygonscan.com/address/{client.contract_address}")
        except Exception as e:
            click.secho(f" [ERROR] Failed to query Amoy status: {e}", fg="red", bold=True)


if __name__ == "__main__":
    cli()
