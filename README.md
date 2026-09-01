# HH-FaceChain: Face -> Social Media Search -> Blockchain Verification Pipeline

A tamper-evident, production-ready pipeline that accepts a facial scan image, performs a genuine runtime reverse-image search across social platforms via Google Lens (with automatic ImgBB image provisioning), extracts post metadata, re-verifies identity using high-precision facial embeddings, and cryptographically anchors the post on a blockchain (Pure-Python Local Simulated Blockchain or Polygon Amoy Testnet).

---

## 1. What It Does

HH-FaceChain provides cryptographic provenance and verification for digital media appearances. Given an input facial photo, the pipeline autonomously proves where that face appears on social media and anchors a tamper-evident record on-chain.

```
+---------------------------------------------------------------------------------------+
|                                    HH-FACECHAIN PIPELINE                              |
+---------------------------------------------------------------------------------------+
                                           |
                                [ Input Face Image ]
                                           |
                                           v
 +-------------------------------------------------------------------------------------+
 | STAGE 1: Face Identification & Extraction                                           |
 | - InsightFace buffalo_l (ONNX Runtime, CPU/CUDA fallback)                           |
 | - Detects largest face by bbox area; generates 512-d L2-normalized embedding        |
 +-------------------------------------------------------------------------------------+
                                           |
                                           v
 +-------------------------------------------------------------------------------------+
 | STAGE 2: Genuine Social Media Search & Re-Verification                              |
 | 1. Upload scan to ImgBB REST API -> Public reachable URL                            |
 | 2. Query SerpApi Google Lens engine (engine=google_lens)                            |
 | 3. Filter candidates to social domains (x.com, instagram.com, reddit.com, etc.)     |
 | 4. Extract post metadata (oEmbed / OpenGraph HTML via BeautifulSoup)                |
 | 5. Download candidate post image & re-verify face match:                            |
 |    Cosine Distance = 1.0 - (u . v) / (||u|| ||v||) < 0.35 (Configurable)            |
 +-------------------------------------------------------------------------------------+
                                           |
                                           v
 +-------------------------------------------------------------------------------------+
 | STAGE 3: Blockchain Canonical Anchoring & Live Verification                         |
 | - Compute deterministic SHA-256 hash over canonical JSON (sort_keys=True)           |
 | - Dual Network Support:                                                             |
 |   * Local Network (--network local): Pure-Python PoW chain (difficulty '000')       |
 |   * Polygon Amoy (--network amoy): PostAnchor.sol smart contract (Chain ID 80002)  |
 | - Live Re-Verification (`verify` command): Re-fetches post and validates on-chain    |
 | - Tamper Detection (`tamper` command): Proves cryptographic failure on mutation     |
 +-------------------------------------------------------------------------------------+
```

---

## 2. Architecture

```
hh-facechain/
├── config/
│   └── contracts.json        # Smart contract ABI, address, and deployment metadata
├── contracts/
│   └── PostAnchor.sol        # Solidity smart contract for immutable anchoring
├── faceid/
│   ├── __init__.py
│   └── encoder.py            # InsightFace buffalo_l wrapper, embeddings & cosine distance
├── search/
│   ├── __init__.py
│   ├── imgbb_client.py       # ImgBB REST API client with retry & rate limiting
│   ├── lens_client.py        # SerpApi Google Lens client
│   ├── post_extractor.py     # Shared oEmbed & OpenGraph metadata extractor
│   └── matcher.py            # Search orchestration, candidate filtering & re-match
├── chain/
│   ├── __init__.py
│   ├── anchor.py             # Canonical JSON record builder and SHA-256 hasher
│   ├── local_chain.py        # Simulated PoW blockchain with JSON persistence
│   └── web3_client.py        # Web3.py client for Polygon Amoy contract interaction
├── scripts/
│   └── deploy.py             # Solc compiler & deployment script for Polygon Amoy
├── tests/
│   ├── fixtures/             # Test face images and negative controls
│   ├── test_faceid.py        # Stage 1 unit tests
│   ├── test_search.py        # Stage 2 unit tests
│   ├── test_chain.py         # Stage 3 unit tests
│   └── test_cli.py           # Integration CLI tests
├── demo/
│   ├── scan1.jpg             # Public figure 1 scan (Barack Obama)
│   └── scan2.jpg             # Public figure 2 scan (Joe Biden)
├── main.py                   # Click-based CLI entrypoint
├── requirements.txt          # Pinned project dependencies
├── .env.example              # Environment variables template
└── README.md                 # Complete documentation and verification records
```

---

## 3. Setup

### Prerequisites
- Python 3.11 or newer (Python 3.13 tested)
- Git

### Installation

1. **Clone or navigate to the repository:**
   ```bash
   git clone https://github.com/your-username/hh-facechain.git
   cd hh-facechain
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   > [!NOTE]
   > On initial run, InsightFace automatically downloads `buffalo_l` ONNX models (~280 MB) to `~/.insightface/models/buffalo_l/`. This is a one-time automatic operation.

4. **Configure Environment Variables:**
   Create a `.env` file from `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Populate the following variables:
   ```ini
   SERPAPI_KEY=your_serpapi_key_here
   IMGBB_KEY=your_imgbb_key_here
   PRIVATE_KEY=your_polygon_amoy_private_key_here
   RPC_URL=https://polygon-amoy-bor-rpc.publicnode.com
   FACE_MATCH_TOL=0.35
   ```

---

## 4. How to Run

### Option A: Localhost Presentation UI (Reviewer-Friendly Console)

Launch the interactive, lab-instrument-style forensic dashboard on localhost:

```bash
python server.py
```
Then open your browser to **[http://localhost:8000](http://localhost:8000)**.

**Features of the Presentation UI (v2.1.0):**
- **▶ 1-Click Guided Demo Mode**: Designed specifically for hackathon judges and code reviewers. Automatically executes the complete 4-act flow on bundled demo figures while streaming real-time plain-English narration and auto-scrolling through each phase.
- **Persistent 4-Act Story Strip**: Horizontal narrative header (`1 · UPLOAD FACE` &rarr; `2 · FIND REAL POST` &rarr; `3 · ANCHOR ON-CHAIN` &rarr; `4 · VERIFY / TAMPER`) with glowing live-stage trackers and smooth click-to-scroll navigation.
- **One Canonical Accuracy Metric**: Single authoritative percentage gauge (`round((1 - cosine_distance) * 100, 1)%`) used consistently across hero readouts, comparator headers, and candidate lists.
- **Single-Column Story Narrative**: Logical top-to-bottom layout:
  1. `THE MATCH`: Hero accuracy gauge + side-by-side face comparator + platform chip + post URL + author + extracted text.
  2. `THE BLOCKCHAIN PROOF`: SHA-256 canonical content hash, transaction/block hash, block index, human-readable UTC timestamp, deployer wallet, and 1-click PolygonScan explorer link.
  3. `TEST IT YOURSELF`: Three prominent action buttons (`Verify this record on-chain`, `Tamper one character`, `See all candidates`).
- **Inline Live Verification & Tamper Lab**: Verification and tamper simulations render directly inline beneath the action buttons (no disruptive modals). Side-by-side comparison displays original vs. live recomputed hash and proves the cryptographic Avalanche Effect.
- **Plain-English Subtitles & ⓘ Tooltips**: Hoverable explanatory tooltips for Cosine Distance, Canonical Content Hash, Smart Contracts, PoW, Avalanche Effect, and OpenGraph metadata.
- **Simplified Clean Sidebar**: Clean primary controls with drag-and-drop file upload, test scan presets, network switcher, and collapsed Advanced Settings accordion (Search Engine, Depth Slider, Till-Success, and Tolerance).
- **Collapsible Monospace Log Console**: Streams stage-by-stage pipeline logs in real time.

---

### Option B: Command-Line Interface (CLI)

#### Command 1: Full End-to-End Pipeline (`run`)

Runs the complete pipeline (Stage 1 Face ID -> Stage 2 Live Search -> Stage 3 Blockchain Anchor):

```bash
python main.py run --image demo/scan1.jpg --network local
```

#### Real Sample Output (Local Network):
```
======================================================================
  HH Goa 2026: Face -> Social Search -> Blockchain Pipeline
======================================================================

[*] Starting Pipeline Execution
    - Input image: demo/scan1.jpg
    - Target network: LOCAL
    - Cosine distance tolerance: 0.35
    - Offline demo mode: False

--- STAGE 1: Face Identification ---
 [OK] Face detected in 2.43s
      - Total faces detected: 1
      - Selected largest face bbox: [374, 81, 607, 432]
      - Detection confidence score: 0.8881
      - Embedding dimensions: 512 (L2-normalized)

--- STAGE 2: Genuine Social Media Search & Face Re-Match ---
 [*] Uploaded scan to ImgBB: https://i.ibb.co/XxgRQJR4/495dc4d570c3.jpg
 [*] Google Lens matches found: 59
 [*] Filtered social candidates: 6

 Candidate Evaluation Logs:
   [1] https://www.instagram.com/p/Dbnq0lYkU3-/ [dist: 0.093] -> ACCEPTED (cosine distance: 0.0930 < 0.35)

 [OK] Verified face match confirmed!
      - Matched Platform: instagram
      - Post URL: https://www.instagram.com/p/Dbnq0lYkU3-/
      - Author: N/A
      - Cosine Distance: 0.0930 (threshold: < 0.35)
      - Image SHA-256: 

--- STAGE 3: Blockchain Anchoring ---
 [*] Canonical Record:
      author: 
      image_sha256: 
      platform: instagram
      post_url: https://www.instagram.com/p/Dbnq0lYkU3-/
      posted_at: 
      text: Instagram
 [*] Deterministic Content Hash (SHA-256): 6f9390ee19226082d9094c39a5a12f6f99f4f20b4a56cf39c68bb0e1eb42a03b

 [OK] Anchored on Local Simulated Blockchain!
      - Block Index: #3
      - Block Hash: 0000941124320cd1f02a3eba3fe3f8afc7e74aaad7d28e85ede075798e20c431
      - Previous Hash: 000d7ef6ed57861d2a742622fed79fb08ac9f307b530933db179c8c927a39e3a
      - Proof-of-Work Nonce: 649
      - Timestamp: 1788237264

=== Execution Complete ===
 [*] Saved verified record to: out\record.json
 [*] Saved post image to: out\post_image.jpg
```

---

### Command 2: Face Search Mode (`search`)

Runs Stages 1 & 2 only, listing all candidate evaluations and distances without blockchain anchoring:

```bash
python main.py search --image demo/scan1.jpg
```

#### Real Sample Output:
```
======================================================================
  HH Goa 2026: Face -> Social Search -> Blockchain Pipeline
======================================================================

[*] Running Face Search Mode (Stages 1 & 2)
    - Input image: demo/scan1.jpg
    - Tolerance: 0.35

 [OK] Face detected: bbox=[374.0, 81.0, 607.0, 432.0], det_score=0.8881
 [*] Google Lens matches: 59
 [*] Social candidates: 6

 Candidate Evaluation:
   [1] https://www.instagram.com/p/Dbnq0lYkU3-/ [dist: 0.093] -> ACCEPTED (cosine distance: 0.0930 < 0.35)

 [ACCEPTED MATCH]
   - Platform: instagram
   - Post URL: https://www.instagram.com/p/Dbnq0lYkU3-/
   - Author: N/A
   - Distance: 0.0930
   - Image SHA256: 
```

---

### Command 3: Live Verification (`verify`)

LIVE re-fetches the post from the web, re-builds the canonical record, recomputes the content hash, and verifies it against the blockchain record:

```bash
python main.py verify --record out/record.json --network local
```

#### Real Sample Output (PASS):
```
======================================================================
  HH Goa 2026: Face -> Social Search -> Blockchain Pipeline
======================================================================

=== LIVE Blockchain Re-Verification ===
 [*] Target Network: LOCAL
 [*] Post URL: https://www.instagram.com/p/Dbnq0lYkU3-/
 [*] Saved Content Hash: 6f9390ee19226082d9094c39a5a12f6f99f4f20b4a56cf39c68bb0e1eb42a03b

 [*] LIVE re-fetching post metadata from web...
 [*] Recomputed Live Content Hash: 6f9390ee19226082d9094c39a5a12f6f99f4f20b4a56cf39c68bb0e1eb42a03b
 [*] Verifying hash against LOCAL blockchain...

 [PASS] Blockchain Verification Successful!
   - Network: LOCAL SIMULATED BLOCKCHAIN
   - Block Index: #3
   - Block Hash: 0000941124320cd1f02a3eba3fe3f8afc7e74aaad7d28e85ede075798e20c431
   - Anchored At: 2026-09-01T04:34:24Z
   - Post URL: https://www.instagram.com/p/Dbnq0lYkU3-/
   - Content Hash: 6f9390ee19226082d9094c39a5a12f6f99f4f20b4a56cf39c68bb0e1eb42a03b
```

---

### Command 4: Tamper-Evidence Demonstration (`tamper`)

Simulates a malicious edit by modifying one character in the cached post content, recalculating the SHA-256 digest, and proving that the on-chain hash rejects the mutated record:

```bash
python main.py tamper --record out/record.json --network local
```

#### Real Sample Output (FAIL / TAMPER DETECTED):
```
======================================================================
  HH Goa 2026: Face -> Social Search -> Blockchain Pipeline
======================================================================

=== Tamper-Evidence Demonstration ===
 [*] Original Post Text: 'Instagram'
 [*] Original Canonical Hash: 6f9390ee19226082d9094c39a5a12f6f99f4f20b4a56cf39c68bb0e1eb42a03b

 [!] Mutating record text to simulate tampering:
     Before: 'Instagram'
     After:  'InstagraX'

 [*] Side-by-Side Hash Comparison:
     Original Hash (On-Chain): 6f9390ee19226082d9094c39a5a12f6f99f4f20b4a56cf39c68bb0e1eb42a03b
     Tampered Hash (Mutated):  616b34a80325afef2264462da79fd362c797b912cc3567fd7161d069bec5a638

 [*] Querying LOCAL blockchain for the tampered hash...

 [FAIL] TAMPERING DETECTED! Content hash mismatch on-chain.
   Reason: Content hash 616b34a80325afef2264462da79fd362c797b912cc3567fd7161d069bec5a638 not found in local blockchain.
 [DEMO SUCCESS] The blockchain rejected the altered record with cryptographic proof.
```

---

### Command 5: Inspect Chain Status (`chain-status`)

```bash
python main.py chain-status --network local
```

#### Real Sample Output:
```
======================================================================
  HH Goa 2026: Face -> Social Search -> Blockchain Pipeline
======================================================================

=== Blockchain Status: LOCAL ===
  network: local
  chain_file: C:\Users\ANSH\.gemini\antigravity\scratch\HHGOATASK3FACE\data\local_chain.json
  total_blocks: 4
  total_anchored_records: 3
  latest_block_index: 3
  latest_block_hash: 0000941124320cd1f02a3eba3fe3f8afc7e74aaad7d28e85ede075798e20c431
  pow_difficulty: 000
  integrity_valid: True
  status_message: Chain integrity verified successfully.

 Recent Blocks:
   Block #0: hash=0005a9de3e8ef6e3... nonce=3996 records=1
   Block #1: hash=000409579eb8166b... nonce=1084 records=1
   Block #2: hash=000d7ef6ed57861d... nonce=213 records=1
   Block #3: hash=0000941124320cd1... nonce=649 records=1
```

---

## 5. Blockchain Used

The pipeline offers dual-mode blockchain support:

### 1. Local Simulated Blockchain (`--network local`, DEFAULT)
- **Zero external dependencies**, operates completely offline with zero setup.
- Persisted to `data/local_chain.json`.
- Implements Proof-of-Work mining with difficulty prefix `000` (3 leading hex zeros).
- Complete cryptographic block linkage: `block_hash = SHA256(index : timestamp : prev_hash : records : nonce)`.
- Verification recursively validates all block hashes and linkage from the Genesis block to the tip.

### 2. Polygon Amoy Testnet (`--network amoy`)
- Deployed Solidity smart contract `PostAnchor.sol` (Solidity `^0.8.24`).
- **Chain ID:** `80002` (Polygon Amoy Testnet)
- **Contract Address:** [`0x80140BdB94D3808CbcD06f79D4fF3Faa3f591362`](https://amoy.polygonscan.com/address/0x80140BdB94D3808CbcD06f79D4fF3Faa3f591362)
- **Deployment Transaction:** [`0x31cfe03140f9cbef1fe9ebd35f707d7d11fe7645a526034edeb6e3f996b5c424`](https://amoy.polygonscan.com/tx/31cfe03140f9cbef1fe9ebd35f707d7d11fe7645a526034edeb6e3f996b5c424)
- **Live Anchored Record Transaction:** [`0x5aa339b02a2b527777b162aee680e4168fb904e3990ddd3d7e2349c828378635`](https://amoy.polygonscan.com/tx/0x5aa339b02a2b527777b162aee680e4168fb904e3990ddd3d7e2349c828378635)
- **Deployment command:** `python scripts/deploy.py`

#### Real Sample Output (Polygon Amoy Anchoring & Verification):
```bash
python main.py run --image demo/scan1.jpg --network amoy
```
```
 [*] Submitting transaction to Polygon Amoy (0x80140BdB94D3808CbcD06f79D4fF3Faa3f591362)...

 [OK] Anchored on Polygon Amoy Testnet!
      - Contract: 0x80140BdB94D3808CbcD06f79D4fF3Faa3f591362
      - Transaction Hash: 0x5aa339b02a2b527777b162aee680e4168fb904e3990ddd3d7e2349c828378635
      - Block Number: 46418861
      - Explorer Link: https://amoy.polygonscan.com/tx/0x5aa339b02a2b527777b162aee680e4168fb904e3990ddd3d7e2349c828378635
```

```bash
python main.py verify --record out/record.json --network amoy
```
```
 [PASS] Blockchain Verification Successful!
   - Network: POLYGON AMOY TESTNET
   - Contract Address: 0x80140BdB94D3808CbcD06f79D4fF3Faa3f591362
   - Anchored By: 0x2De14c641A9B8c6570Ac6633cF9983dD981C519f
   - Anchored Timestamp: 2026-09-01T04:35:03Z
   - Post URL: https://www.instagram.com/p/Dbnq0lYkU3-/
   - Content Hash: 6f9390ee19226082d9094c39a5a12f6f99f4f20b4a56cf39c68bb0e1eb42a03b
   - Explorer Link: https://amoy.polygonscan.com/address/0x80140BdB94D3808CbcD06f79D4fF3Faa3f591362
```

---

## 6. Proof of Genuine Search

To prove that the pipeline performs a true runtime search (rather than using pre-picked or static values), we evaluate two distinct public figure scans (`demo/scan1.jpg` and `demo/scan2.jpg`):

### Comparison: Scan 1 vs Scan 2

| Metric | Input Scan 1 (`demo/scan1.jpg` - Barack Obama) | Input Scan 2 (`demo/scan2.jpg` - Joe Biden) |
| :--- | :--- | :--- |
| **Stage 1 Detection** | BBox: `[374, 81, 607, 432]`, Confidence: `0.8881` | BBox: `[421, 158, 706, 562]`, Confidence: `0.7989` |
| **Lens Visual Matches** | 59 visual matches returned | 59 visual matches returned |
| **Social Candidates Filtered** | 6 candidate URLs (`instagram.com`, `facebook.com`, `reddit.com`) | 2 candidate URLs (`pinterest.com`, `facebook.com`) |
| **Top Candidate 1 Distance** | `https://www.instagram.com/p/Dbnq0lYkU3-/` -> **dist: 0.0930** | `https://www.pinterest.com/...` -> **dist: 0.6840** |
| **Top Candidate 2 Distance** | N/A (Accepted candidate 1) | `https://www.facebook.com/...` -> **dist: 0.4945** |
| **Match Outcome** | **ACCEPTED (Match Confirmed)** | **REJECTED (Threshold Protected)** |
| **Anchored Content Hash** | `6f9390ee19226082d9094c39a5a12f6f99f4f20b4a56cf39c68bb0e1eb42a03b` | None (Clean zero-match exit) |

This demonstrates that every run dynamically inspects Google Lens visual matches, extracts candidates, encodes retrieved images in real-time, and strictly applies the cosine distance threshold.

---

## 7. Known Limitations

1. **Facial Pose & Lighting Variance:** Extreme side angles (>60 degree yaw) or heavy occlusions (e.g. sunglasses) reduce detection confidence and increase cosine distance.
2. **Platform Anti-Scraping / Dynamic SPAs:** Modern platforms (e.g. Instagram, Facebook) increasingly gate post HTML behind client-side JavaScript rendering or login walls. When `og:image` tags are hidden, candidate verification gracefully falls back to visual search match thumbnails.
3. **Third-Party API Availability:** Runtime search relies on SerpApi (Google Lens engine) and ImgBB. If API quotas are exceeded, the pipeline provides an `--offline-demo` mode for offline unit testing.
4. **Post Deletions & Modifications:** If a social media post is subsequently edited or deleted after anchoring, the `verify` command will correctly detect that the live content hash no longer matches the on-chain anchor, producing a `FAIL` (which accurately indicates tampering or modification).
5. **Key Management:** The CLI uses local environment variables (`PRIVATE_KEY`) for testnet deployment. For enterprise production, a hardware security module (HSM) or multi-sig wallet should be used.

---

## 8. Ethics Statement

- **Consent & Scope:** This pipeline is intended strictly for verifying authorized media provenance and public figure appearances. It must not be deployed for non-consensual surveillance, stalking, or harassment.
- **Privacy Preservation:** The blockchain anchors only deterministic 32-byte cryptographic hashes (`bytes32`) and public post URLs. No raw biometric data, embeddings, or personally identifiable information (PII) are stored on the public blockchain.
- **Polite Crawling:** All HTTP requests adhere to rate limiting (>=1s delays between candidate requests), declare standard User-Agents, and respect standard timeout policies.

---

## 9. Assumptions

1. **Face Selection:** When an input image contains multiple faces, the pipeline selects the largest face by bounding box area (`(x2-x1)*(y2-y1)`), assuming the primary subject is in the foreground.
2. **Embedding Normalization:** InsightFace `buffalo_l` provides 512-dimensional L2-normalized embeddings. The cosine distance metric $1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$ is bounded in $[0.0, 2.0]$.
3. **Canonical Determinism:** Only immutable post attributes (`platform`, `post_url`, `author`, `text`, `posted_at`, `image_sha256`) are hashed into the blockchain record. Volatile retrieval timestamps are excluded to ensure deterministic re-verification.
4. **Testnet Gas:** The Polygon Amoy testnet is used for smart contract demonstration; testnet POL has no real-world monetary value.

---

## 10. Verification Protocol & Test Results

Run all unit and integration tests with pytest:
```bash
pytest -v
```

```
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\ANSH\.gemini\antigravity\scratch\HHGOATASK3FACE
collected 17 items

tests/test_chain.py::test_canonical_record_determinism PASSED            [  5%]
tests/test_chain.py::test_hash_to_bytes32 PASSED                         [ 11%]
tests/test_chain.py::test_local_blockchain_pow_and_verification PASSED   [ 17%]
tests/test_chain.py::test_local_blockchain_tamper_detection PASSED       [ 23%]
tests/test_cli.py::test_cli_help PASSED                                  [ 29%]
tests/test_cli.py::test_cli_chain_status_local PASSED                    [ 35%]
tests/test_cli.py::test_cli_run_verify_tamper_offline_flow PASSED        [ 41%]
tests/test_faceid.py::test_cosine_distance_properties PASSED             [ 47%]
tests/test_faceid.py::test_same_person_matches PASSED                    [ 52%]
tests/test_faceid.py::test_different_people_do_not_match PASSED          [ 58%]
tests/test_faceid.py::test_no_face_found_raises_exception PASSED         [ 64%]
tests/test_faceid.py::test_invalid_image_raises_error PASSED             [ 70%]
tests/test_faceid.py::test_encode_face_with_meta PASSED                  [ 76%]
tests/test_search.py::test_social_platform_domain_filtering PASSED       [ 82%]
tests/test_search.py::test_url_normalization PASSED                      [ 88%]
tests/test_search.py::test_opengraph_extraction_parser PASSED            [ 94%]
tests/test_search.py::test_offline_demo_matcher PASSED                   [100%]

======================= 17 passed, 4 warnings in 7.56s ========================
```

---

## 11. License

This project is licensed under the [MIT License](LICENSE).
