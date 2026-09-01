"""Web3 client for Polygon Amoy blockchain anchoring and verification."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from chain.anchor import hash_to_bytes32

load_dotenv()
load_dotenv(".env.txt")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "contracts.json")

FALLBACK_RPCS = [
    os.getenv("RPC_URL", "https://polygon-amoy-bor-rpc.publicnode.com"),
    "https://polygon-amoy-bor-rpc.publicnode.com",
    "https://polygon-amoy.drpc.org",
    "https://rpc-amoy.polygon.technology",
]


class Web3ChainError(Exception):
    """Raised when a Web3 operation or RPC call fails."""
    pass


class Web3Client:
    """Interacts with the PostAnchor smart contract on Polygon Amoy."""

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.w3, self.active_rpc = self._get_connection()
        self.contract_address, self.abi = self._load_contract_config()
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.abi)

    def _get_connection(self) -> tuple[Web3, str]:
        for rpc in FALLBACK_RPCS:
            if not rpc:
                continue
            try:
                w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 15}))
                if w3.is_connected():
                    try:
                        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                    except Exception:
                        pass
                    return w3, rpc
            except Exception:
                continue
        raise Web3ChainError("Failed to connect to any Polygon Amoy RPC endpoint.")

    def _load_contract_config(self) -> tuple[str, list]:
        if not os.path.exists(self.config_path):
            raise Web3ChainError(
                f"Contract config file not found at {self.config_path}. "
                "Run 'python scripts/deploy.py' to deploy the contract first."
            )
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        addr = data.get("address")
        abi = data.get("abi")
        if not addr or not abi:
            raise Web3ChainError("Invalid contract configuration in config/contracts.json.")
        return Web3.to_checksum_address(addr), abi

    def anchor(self, content_hash: str, post_url: str, timeout: int = 120) -> dict[str, Any]:
        """
        Submits an on-chain transaction to anchor content_hash (bytes32) and post_url.
        """
        priv_key = os.getenv("PRIVATE_KEY")
        if not priv_key:
            raise Web3ChainError("PRIVATE_KEY environment variable is required to anchor on Amoy.")
        if not priv_key.startswith("0x"):
            priv_key = "0x" + priv_key

        account = self.w3.eth.account.from_key(priv_key)
        sender = account.address
        content_hash_bytes = hash_to_bytes32(content_hash)

        # Check if already anchored on-chain
        try:
            existing = self.contract.functions.anchors(content_hash_bytes).call()
            if existing[0] > 0:  # anchoredAt > 0
                raise Web3ChainError(
                    f"Content hash {content_hash} is already anchored on-chain at timestamp {existing[0]} by {existing[1]}."
                )
        except Web3ChainError:
            raise
        except Exception:
            pass

        chain_id = self.w3.eth.chain_id
        nonce = self.w3.eth.get_transaction_count(sender, "pending")
        gas_price = self.w3.eth.gas_price

        # Build transaction
        tx_call = self.contract.functions.anchor(content_hash_bytes, post_url)
        try:
            estimated_gas = tx_call.estimate_gas({"from": sender})
            gas_limit = int(estimated_gas * 1.3)
        except Exception:
            gas_limit = 250000

        tx = tx_call.build_transaction({
            "chainId": chain_id,
            "from": sender,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": int(gas_price * 1.2),
        })

        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=priv_key)
        tx_hash_bytes = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_hash_hex = tx_hash_bytes.hex()
        if not tx_hash_hex.startswith("0x"):
            tx_hash_hex = "0x" + tx_hash_hex

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash_bytes, timeout=timeout)
        if receipt.status != 1:
            raise Web3ChainError(f"Transaction reverted on-chain (tx: {tx_hash_hex})")

        return {
            "network": "amoy",
            "contract_address": self.contract_address,
            "tx_hash": tx_hash_hex,
            "block_number": receipt.blockNumber,
            "content_hash": content_hash.lower(),
            "post_url": post_url,
            "sender": sender,
            "explorer_url": f"https://amoy.polygonscan.com/tx/{tx_hash_hex}",
            "contract_explorer_url": f"https://amoy.polygonscan.com/address/{self.contract_address}",
        }

    def verify_on_chain(self, content_hash: str, expected_post_url: str | None = None) -> tuple[bool, dict[str, Any] | None, str]:
        """
        Queries the smart contract mapping anchors(bytes32) to verify registration.
        """
        content_hash_bytes = hash_to_bytes32(content_hash)
        try:
            anchor_tuple = self.contract.functions.anchors(content_hash_bytes).call()
            anchored_at, by_address, stored_post_url = anchor_tuple

            if anchored_at == 0:
                return False, None, f"Content hash {content_hash} has not been anchored on Polygon Amoy."

            record = {
                "network": "amoy",
                "contract_address": self.contract_address,
                "content_hash": content_hash.lower(),
                "anchored_at": anchored_at,
                "anchored_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(anchored_at)),
                "by": by_address,
                "post_url": stored_post_url,
            }

            if expected_post_url and stored_post_url != expected_post_url:
                return False, record, (
                    f"Hash verified on-chain, but post URL mismatch: "
                    f"stored='{stored_post_url}' vs expected='{expected_post_url}'"
                )

            return True, record, (
                f"Anchored on Polygon Amoy at block timestamp {anchored_at} "
                f"({record['anchored_at_iso']}) by {by_address}."
            )

        except Exception as exc:
            return False, None, f"Failed to query smart contract: {exc}"

    def get_status(self) -> dict[str, Any]:
        """Returns connection and contract status."""
        connected = self.w3.is_connected()
        latest_block = None
        if connected:
            try:
                latest_block = self.w3.eth.block_number
            except Exception:
                pass
        return {
            "network": "Polygon Amoy Testnet",
            "chain_id": 80002,
            "contract_address": self.contract_address,
            "active_rpc": self.active_rpc,
            "connected": connected,
            "latest_block": latest_block,
            "wallet_address": self.account.address if self.account else None,
            "status_message": "Connected to Polygon Amoy testnet." if connected else "Disconnected.",
        }


# Backward compatibility aliases
PolygonAmoyClient = Web3Client
