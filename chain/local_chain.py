"""Pure-Python simulated local blockchain with Proof-of-Work and JSON persistence."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

DEFAULT_CHAIN_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "local_chain.json")
POW_DIFFICULTY_PREFIX = "000"  # 3 leading zero hex chars


class LocalChainError(Exception):
    """Raised when an operation on the local simulated blockchain fails."""
    pass


class Block:
    """Represents a single block in the local simulated blockchain."""

    def __init__(
        self,
        index: int,
        timestamp: float,
        prev_hash: str,
        records: list[dict[str, Any]],
        nonce: int = 0,
        block_hash: str = "",
    ):
        self.index = index
        self.timestamp = timestamp
        self.prev_hash = prev_hash
        self.records = records
        self.nonce = nonce
        self.block_hash = block_hash or self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculates deterministic SHA-256 hash of block contents."""
        records_str = json.dumps(self.records, sort_keys=True, ensure_ascii=False)
        payload = f"{self.index}:{self.timestamp:.4f}:{self.prev_hash}:{records_str}:{self.nonce}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def mine(self, prefix: str = POW_DIFFICULTY_PREFIX) -> None:
        """Simple Proof-of-Work: finds nonce yielding block hash with required leading zeros."""
        while True:
            current_hash = self.calculate_hash()
            if current_hash.startswith(prefix):
                self.block_hash = current_hash
                break
            self.nonce += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
            "records": self.records,
            "nonce": self.nonce,
            "block_hash": self.block_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Block:
        return cls(
            index=int(data["index"]),
            timestamp=float(data["timestamp"]),
            prev_hash=str(data["prev_hash"]),
            records=list(data["records"]),
            nonce=int(data["nonce"]),
            block_hash=str(data["block_hash"]),
        )


class LocalBlockchain:
    """Manages the simulated blockchain state and persistence."""

    def __init__(self, filepath: str | None = None):
        self.filepath = filepath or os.getenv("LOCAL_CHAIN_FILE", DEFAULT_CHAIN_FILE)
        self.chain: list[Block] = []
        self._load_or_init()

    def _create_genesis_block(self) -> Block:
        genesis = Block(
            index=0,
            timestamp=1700000000.0,
            prev_hash="0" * 64,
            records=[{"genesis": "HH Goa 2026 FaceChain Genesis Block"}],
            nonce=0,
        )
        genesis.mine(POW_DIFFICULTY_PREFIX)
        return genesis

    def _load_or_init(self) -> None:
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chain = [Block.from_dict(b) for b in data]
                if not self.chain:
                    self._reset_to_genesis()
            except Exception:
                self._reset_to_genesis()
        else:
            self._reset_to_genesis()

    def _reset_to_genesis(self) -> None:
        self.chain = [self._create_genesis_block()]
        self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump([b.to_dict() for b in self.chain], f, indent=2, ensure_ascii=False)

    @property
    def last_block(self) -> Block:
        return self.chain[-1]

    def anchor(self, content_hash: str, post_url: str, allow_existing: bool = True) -> dict[str, Any]:
        """
        Anchors a content hash and post URL in a new mined block.
        If already anchored and allow_existing=True, returns existing block record.
        """
        clean_hash = content_hash.lower()

        # Check for existing anchor
        for block in self.chain:
            for rec in block.records:
                if rec.get("content_hash") == clean_hash:
                    if allow_existing:
                        return {
                            "network": "local",
                            "block_index": block.index,
                            "block_hash": block.block_hash,
                            "prev_hash": block.prev_hash,
                            "nonce": block.nonce,
                            "content_hash": clean_hash,
                            "post_url": rec.get("post_url", post_url),
                            "timestamp": block.timestamp,
                            "anchored_at_iso": rec.get("anchored_at_iso") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(block.timestamp)),
                            "already_anchored": True,
                            "status_label": f"Local PoW Block #{block.index} (Existing)",
                        }
                    raise LocalChainError(
                        f"Content hash {clean_hash} is already anchored in block #{block.index}."
                    )

        now = time.time()
        record = {
            "content_hash": clean_hash,
            "post_url": post_url,
            "anchored_at": int(now),
            "anchored_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }

        new_block = Block(
            index=self.last_block.index + 1,
            timestamp=now,
            prev_hash=self.last_block.block_hash,
            records=[record],
            nonce=0,
        )
        new_block.mine(POW_DIFFICULTY_PREFIX)

        self.chain.append(new_block)
        self._save()

        return {
            "network": "local",
            "block_index": new_block.index,
            "block_hash": new_block.block_hash,
            "prev_hash": new_block.prev_hash,
            "nonce": new_block.nonce,
            "content_hash": clean_hash,
            "post_url": post_url,
            "anchored_at": record["anchored_at"],
            "total_blocks": len(self.chain),
        }

    def verify_chain_integrity(self) -> tuple[bool, str]:
        """Validates all block hashes, PoW difficulties, and linkage."""
        for i in range(len(self.chain)):
            block = self.chain[i]
            recomputed = block.calculate_hash()

            if block.block_hash != recomputed:
                return False, f"Block #{block.index} hash mismatch (stored: {block.block_hash}, recomputed: {recomputed})"

            if not block.block_hash.startswith(POW_DIFFICULTY_PREFIX):
                return False, f"Block #{block.index} failed PoW difficulty prefix '{POW_DIFFICULTY_PREFIX}'"

            if i > 0:
                prev_block = self.chain[i - 1]
                if block.prev_hash != prev_block.block_hash:
                    return False, f"Block #{block.index} prev_hash broken ({block.prev_hash} != {prev_block.block_hash})"

        return True, "Chain integrity verified successfully."

    def verify(self, content_hash: str, expected_post_url: str | None = None) -> tuple[bool, dict[str, Any] | None, str]:
        """
        Walks the chain, checks full integrity, and searches for the anchored content hash.

        Returns:
            tuple[bool, dict | None, str]: (passed, anchor_record, message)
        """
        is_intact, err_msg = self.verify_chain_integrity()
        if not is_intact:
            return False, None, f"Local blockchain corrupt: {err_msg}"

        clean_hash = content_hash.lower()

        for block in self.chain:
            for rec in block.records:
                if rec.get("content_hash") == clean_hash:
                    if expected_post_url and rec.get("post_url") != expected_post_url:
                        return False, rec, f"Hash matched but post_url mismatch: {rec.get('post_url')} != {expected_post_url}"

                    return True, {
                        "network": "local",
                        "block_index": block.index,
                        "block_hash": block.block_hash,
                        "content_hash": clean_hash,
                        "post_url": rec.get("post_url"),
                        "anchored_at": rec.get("anchored_at"),
                        "anchored_at_iso": rec.get("anchored_at_iso"),
                    }, f"Found verified anchor in block #{block.index}."

        return False, None, f"Content hash {clean_hash} not found in local blockchain."

    def get_status(self) -> dict[str, Any]:
        """Returns overview status of the local blockchain."""
        is_valid, msg = self.verify_chain_integrity()
        total_records = sum(len(b.records) for b in self.chain if b.index > 0)
        return {
            "network": "local",
            "chain_file": self.filepath,
            "total_blocks": len(self.chain),
            "total_anchored_records": total_records,
            "latest_block_index": self.last_block.index,
            "latest_block_hash": self.last_block.block_hash,
            "pow_difficulty": POW_DIFFICULTY_PREFIX,
            "integrity_valid": is_valid,
            "status_message": msg,
        }
