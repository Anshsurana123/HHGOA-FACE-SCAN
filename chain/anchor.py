"""Canonical record builder and deterministic hashing for blockchain anchoring."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def make_canonical_record(post_dict: dict[str, Any]) -> dict[str, str]:
    """
    Creates a canonical record dictionary containing only immutable post fields.
    Volatile fields (such as retrieved_at) are explicitly excluded for deterministic re-verification.
    """
    return {
        "author": str(post_dict.get("author", "") or ""),
        "image_sha256": str(post_dict.get("image_sha256", "") or ""),
        "platform": str(post_dict.get("platform", "") or ""),
        "post_url": str(post_dict.get("post_url", "") or ""),
        "posted_at": str(post_dict.get("posted_at", "") or ""),
        "text": str(post_dict.get("text", "") or ""),
    }


def compute_record_hash(record: dict[str, Any]) -> str:
    """
    Computes deterministic SHA-256 hex digest of a canonical record.

    JSON serialization uses:
        - sort_keys=True
        - ensure_ascii=False
        - UTF-8 encoding
    """
    canonical = make_canonical_record(record)
    canonical_json = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def hash_to_bytes32(content_hash_hex: str) -> bytes:
    """
    Converts a 64-character SHA-256 hex string to 32 raw bytes for smart contract calls.
    """
    clean_hex = content_hash_hex.lower()
    if clean_hex.startswith("0x"):
        clean_hex = clean_hex[2:]
    if len(clean_hex) != 64:
        raise ValueError(f"Expected 64 hex characters (32 bytes), got {len(clean_hex)}")
    return bytes.fromhex(clean_hex)
