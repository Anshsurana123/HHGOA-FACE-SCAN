"""Unit tests for Stage 3: Blockchain canonical hashing and local simulated blockchain."""

import os
import json
import tempfile
import pytest

from chain.anchor import make_canonical_record, compute_record_hash, hash_to_bytes32
from chain.local_chain import LocalBlockchain, LocalChainError, POW_DIFFICULTY_PREFIX


def test_canonical_record_determinism():
    """Validates deterministic serialization and exclusion of volatile fields."""
    post_a = {
        "platform": "x",
        "post_url": "https://x.com/user/status/100",
        "author": "Alice",
        "text": "Hello Blockchain!",
        "image_sha256": "abc123def456",
        "posted_at": "2026-01-01T00:00:00Z",
        "retrieved_at": "2026-03-01T12:34:56Z",  # Volatile field
        "session_id": 9999,                      # Volatile field
    }

    post_b = {
        "text": "Hello Blockchain!",
        "author": "Alice",
        "platform": "x",
        "posted_at": "2026-01-01T00:00:00Z",
        "image_sha256": "abc123def456",
        "post_url": "https://x.com/user/status/100",
        "retrieved_at": "2026-09-09T99:99:99Z",  # Different volatile field
    }

    canonical_a = make_canonical_record(post_a)
    canonical_b = make_canonical_record(post_b)

    assert "retrieved_at" not in canonical_a
    assert "session_id" not in canonical_a
    assert canonical_a == canonical_b

    hash_a = compute_record_hash(post_a)
    hash_b = compute_record_hash(post_b)

    assert len(hash_a) == 64
    assert hash_a == hash_b


def test_hash_to_bytes32():
    """Validates conversion of 64-char hex string to 32 raw bytes."""
    hex_str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    raw_bytes = hash_to_bytes32(hex_str)
    assert isinstance(raw_bytes, bytes)
    assert len(raw_bytes) == 32
    assert raw_bytes.hex() == hex_str

    # Handles leading 0x
    assert hash_to_bytes32("0x" + hex_str) == raw_bytes

    # Rejects invalid lengths
    with pytest.raises(ValueError):
        hash_to_bytes32("short_hex")


def test_local_blockchain_pow_and_verification():
    """Validates mining, block linkage, and verification on the simulated local chain."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chain_path = os.path.join(tmpdir, "test_chain.json")
        chain = LocalBlockchain(filepath=chain_path)

        assert len(chain.chain) == 1  # Genesis block
        assert chain.chain[0].index == 0
        assert chain.chain[0].block_hash.startswith(POW_DIFFICULTY_PREFIX)

        # Anchor a record
        content_hash = "1111111111111111111111111111111111111111111111111111111111111111"
        post_url = "https://x.com/test/status/1"
        anchor_res = chain.anchor(content_hash, post_url)

        assert anchor_res["block_index"] == 1
        assert anchor_res["block_hash"].startswith(POW_DIFFICULTY_PREFIX)
        assert anchor_res["prev_hash"] == chain.chain[0].block_hash
        assert len(chain.chain) == 2

        # Verify anchored record
        is_valid, record, msg = chain.verify(content_hash, expected_post_url=post_url)
        assert is_valid is True
        assert record is not None
        assert record["block_index"] == 1

        # Duplicate anchor with allow_existing=True returns existing proof
        existing_res = chain.anchor(content_hash, post_url, allow_existing=True)
        assert existing_res.get("already_anchored") is True
        assert existing_res["block_index"] == 1

        # Duplicate anchor with allow_existing=False raises LocalChainError
        with pytest.raises(LocalChainError):
            chain.anchor(content_hash, post_url, allow_existing=False)

        # Unanchored hash returns is_valid=False
        unanchored = "2222222222222222222222222222222222222222222222222222222222222222"
        is_val, rec, msg = chain.verify(unanchored)
        assert is_val is False
        assert rec is None


def test_local_blockchain_tamper_detection():
    """Validates that modifying data inside the chain file triggers integrity failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chain_path = os.path.join(tmpdir, "test_chain.json")
        chain = LocalBlockchain(filepath=chain_path)

        content_hash = "3333333333333333333333333333333333333333333333333333333333333333"
        chain.anchor(content_hash, "https://x.com/tamper/test")

        # Mutate saved JSON on disk
        with open(chain_path, "r") as f:
            data = json.load(f)

        data[1]["records"][0]["post_url"] = "https://x.com/tampered_url"

        with open(chain_path, "w") as f:
            json.dump(data, f)

        # Reload chain and verify integrity
        tampered_chain = LocalBlockchain(filepath=chain_path)
        is_intact, msg = tampered_chain.verify_chain_integrity()
        assert is_intact is False
        assert "mismatch" in msg


def test_web3_client_verify_interface():
    """Validates that Web3Client implements both verify and verify_on_chain uniformly."""
    from chain.web3_client import Web3Client
    assert hasattr(Web3Client, "verify")
    assert hasattr(Web3Client, "verify_on_chain")
    assert hasattr(Web3Client, "anchor")
