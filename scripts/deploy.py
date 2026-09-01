"""Deploy PostAnchor smart contract to Polygon Amoy testnet."""

from __future__ import annotations

import json
import os
import sys
import time
from dotenv import load_dotenv
import solcx
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

# Load environment variables
load_dotenv()
load_dotenv(".env.txt")

CONTRACT_SOURCE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "contracts", "PostAnchor.sol")
CONFIG_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "contracts.json")

FALLBACK_RPCS = [
    os.getenv("RPC_URL", "https://polygon-amoy-bor-rpc.publicnode.com"),
    "https://polygon-amoy-bor-rpc.publicnode.com",
    "https://polygon-amoy.drpc.org",
    "https://rpc-amoy.polygon.technology",
]


def get_web3_connection() -> tuple[Web3, str]:
    """Connects to Polygon Amoy using configured or fallback RPCs."""
    for rpc in FALLBACK_RPCS:
        if not rpc:
            continue
        try:
            w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 15}))
            if w3.is_connected():
                # Inject POA middleware if necessary for Polygon Amoy
                try:
                    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                except Exception:
                    pass
                return w3, rpc
        except Exception:
            continue

    raise ConnectionError("Failed to connect to any Polygon Amoy RPC endpoint.")


def compile_contract() -> tuple[dict, list]:
    """Installs solc if needed and compiles PostAnchor.sol."""
    solc_version = "0.8.24"
    installed_versions = [str(v) for v in solcx.get_installed_solc_versions()]
    if solc_version not in installed_versions:
        print(f"[*] Installing Solidity compiler solc {solc_version}...")
        solcx.install_solc(solc_version)

    solcx.set_solc_version(solc_version)

    with open(CONTRACT_SOURCE_PATH, "r", encoding="utf-8") as f:
        source = f.read()

    compiled = solcx.compile_source(
        source,
        output_values=["abi", "bin"],
        solc_version=solc_version,
    )

    contract_id = "<stdin>:PostAnchor"
    if contract_id not in compiled:
        # Fallback search for PostAnchor
        for k in compiled:
            if "PostAnchor" in k:
                contract_id = k
                break

    contract_interface = compiled[contract_id]
    abi = contract_interface["abi"]
    bytecode = contract_interface["bin"]
    return abi, bytecode


def deploy() -> dict:
    """Deploys PostAnchor to Polygon Amoy testnet and saves config."""
    print("=== Deploying PostAnchor to Polygon Amoy Testnet ===")

    priv_key = os.getenv("PRIVATE_KEY")
    if not priv_key:
        print("[ERROR] PRIVATE_KEY is not set in environment or .env file.")
        sys.exit(1)

    if not priv_key.startswith("0x"):
        priv_key = "0x" + priv_key

    w3, active_rpc = get_web3_connection()
    account = w3.eth.account.from_key(priv_key)
    sender = account.address

    balance_wei = w3.eth.get_balance(sender)
    balance_pol = w3.from_wei(balance_wei, "ether")
    chain_id = w3.eth.chain_id

    print(f"[*] Connected to RPC: {active_rpc}")
    print(f"[*] Chain ID: {chain_id}")
    print(f"[*] Deployer Address: {sender}")
    print(f"[*] Balance: {balance_pol} POL")

    if balance_wei == 0:
        print("[ERROR] Deployer wallet has 0 POL balance. Please fund testnet account.")
        sys.exit(1)

    print("[*] Compiling PostAnchor.sol...")
    abi, bytecode = compile_contract()

    PostAnchor = w3.eth.contract(abi=abi, bytecode=bytecode)

    nonce = w3.eth.get_transaction_count(sender, "pending")
    gas_price = w3.eth.gas_price

    # Estimate deployment gas
    estimated_gas = PostAnchor.constructor().estimate_gas({"from": sender})
    gas_limit = int(estimated_gas * 1.3)

    print(f"[*] Building deployment transaction (nonce: {nonce}, gasLimit: {gas_limit})...")
    tx = PostAnchor.constructor().build_transaction({
        "chainId": chain_id,
        "from": sender,
        "nonce": nonce,
        "gas": gas_limit,
        "gasPrice": int(gas_price * 1.2),
    })

    signed_tx = w3.eth.account.sign_transaction(tx, private_key=priv_key)
    print("[*] Broadcasting transaction to Polygon Amoy...")
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    tx_hex = tx_hash.hex()
    print(f"[*] Transaction sent: {tx_hex}")
    print(f"[*] View Tx on Polygonscan: https://amoy.polygonscan.com/tx/{tx_hex}")

    print("[*] Waiting for transaction receipt...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt.status != 1:
        print(f"[ERROR] Deployment transaction failed with status {receipt.status}")
        sys.exit(1)

    contract_address = receipt.contractAddress
    print(f"\n[SUCCESS] PostAnchor deployed successfully!")
    print(f"[*] Contract Address: {contract_address}")
    print(f"[*] Block Number: {receipt.blockNumber}")
    print(f"[*] Polygonscan Link: https://amoy.polygonscan.com/address/{contract_address}")

    deployment_data = {
        "address": contract_address,
        "abi": abi,
        "network": "amoy",
        "chainId": chain_id,
        "deployer": sender,
        "deploymentTx": tx_hex,
        "blockNumber": receipt.blockNumber,
        "deployedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "explorerUrl": f"https://amoy.polygonscan.com/address/{contract_address}",
    }

    os.makedirs(os.path.dirname(CONFIG_OUTPUT_PATH), exist_ok=True)
    with open(CONFIG_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(deployment_data, f, indent=2)

    print(f"[*] Saved configuration to {CONFIG_OUTPUT_PATH}\n")
    return deployment_data


if __name__ == "__main__":
    deploy()
