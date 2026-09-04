"""Unit and integration tests for Agentic OSINT discovery, evidence graph, and tools."""

from __future__ import annotations

import json
import os
import pytest
import numpy as np

from agent.state import (
    ResearchState,
    EvidenceGraph,
    EvidenceNode,
    EvidenceEdge,
    normalize_search_query,
    normalize_target_url,
)
from agent.schemas import AGENT_TOOLS_SCHEMA, get_gemini_tools_declaration, get_openai_tools_declaration
from agent.tools import (
    execute_analyze_image,
    execute_extract_ocr,
    execute_web_search,
    execute_open_url,
    execute_extract_page_images,
    execute_download_candidate_image,
    execute_face_match,
    execute_inspect_candidate_post,
    dispatch_tool,
)
from agent.llm_client import ToolCall, HeuristicAgentClient, get_llm_client
from agent.researcher import ResearchAgent, AgentResult
from chain.anchor import make_canonical_record, compute_record_hash
from chain.local_chain import LocalBlockchain


@pytest.fixture
def sample_image_bytes():
    fixture_path = "tests/fixtures/person1_a.jpg"
    with open(fixture_path, "rb") as f:
        return f.read()


@pytest.fixture
def sample_image_bytes_b():
    fixture_path = "tests/fixtures/person1_b.jpg"
    with open(fixture_path, "rb") as f:
        return f.read()


@pytest.fixture
def dummy_embedding():
    emb = np.random.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


# 1. State Initialization Test
def test_state_initialization(sample_image_bytes, dummy_embedding):
    state = ResearchState(
        input_image_bytes=sample_image_bytes,
        target_embedding=dummy_embedding,
        tolerance=0.35,
        max_iterations=10,
        max_queries=20,
        max_pages=15,
        max_images=50,
    )
    assert state.max_iterations == 10
    assert state.max_queries == 20
    assert state.max_pages == 15
    assert state.max_candidate_images == 50
    assert state.iteration == 0
    assert state.total_queries == 0
    assert state.total_pages_opened == 0
    assert len(state.evidence_graph.nodes) >= 1
    assert state.root_node.node_type == "image"


# 2. Query Deduplication Test
def test_query_deduplication(sample_image_bytes, dummy_embedding):
    state = ResearchState(input_image_bytes=sample_image_bytes, target_embedding=dummy_embedding)
    
    q1 = "Hacker House Goa event"
    q2 = "hacker house goa event "
    q3 = "HACKER   HOUSE   GOA   EVENT"
    
    assert state.is_query_seen(q1) is False
    state.mark_query_seen(q1)
    assert state.is_query_seen(q1) is True
    assert state.is_query_seen(q2) is True
    assert state.is_query_seen(q3) is True
    assert state.total_queries == 1


# 3. URL Deduplication and Normalization Test
def test_url_deduplication(sample_image_bytes, dummy_embedding):
    state = ResearchState(input_image_bytes=sample_image_bytes, target_embedding=dummy_embedding)
    
    u1 = "https://x.com/user/status/12345?utm_source=twitter&ref_src=twsrc"
    u2 = "https://x.com/user/status/12345/"
    u3 = "https://X.COM/user/status/12345"
    
    assert state.is_url_visited(u1) is False
    state.mark_url_visited(u1)
    assert state.is_url_visited(u1) is True
    assert state.is_url_visited(u2) is True
    assert state.is_url_visited(u3) is True
    assert state.total_pages_opened == 1


# 4. Evidence Graph Creation and JSON Serialization Test
def test_evidence_graph_creation_and_serialization():
    graph = EvidenceGraph()
    n1 = graph.add_node("image", "Input Photo", {"bytes": 1000})
    n2 = graph.add_node("ocr", "OCR: Tech Conf", {"text": "Tech Conf"}, parent_id=n1.node_id, relation="extracted_text")
    n3 = graph.add_node("query", "Search: Tech Conf", {"query": "Tech Conf"}, parent_id=n2.node_id, relation="queried_web")
    n4 = graph.add_node("match", "Match Found", {"distance": 0.18}, parent_id=n3.node_id, relation="biometrically_verified")
    
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3
    
    chain = graph.get_chain_to_node(n4.node_id)
    assert len(chain) == 4
    assert chain[0]["id"] == n1.node_id
    assert chain[3]["id"] == n4.node_id
    
    d = graph.to_dict()
    assert d["total_nodes"] == 4
    assert d["total_edges"] == 3
    json_str = json.dumps(d)
    assert "Tech Conf" in json_str


# 5. Tool Schema Formats Test
def test_tool_schemas():
    assert len(AGENT_TOOLS_SCHEMA) >= 10
    gemini_decl = get_gemini_tools_declaration()
    assert "function_declarations" in gemini_decl[0]
    openai_decl = get_openai_tools_declaration()
    assert openai_decl[0]["type"] == "function"


# 6. Malformed LLM Action Handling Test
def test_malformed_tool_action(sample_image_bytes, dummy_embedding):
    state = ResearchState(input_image_bytes=sample_image_bytes, target_embedding=dummy_embedding)
    
    # Non-existent tool call
    res = dispatch_tool(state, "arbitrary_eval_code", {"code": "import os"})
    assert res["status"] == "error"
    assert "not an allowed action" in res["error"]
    
    # Missing parameters
    res_web = dispatch_tool(state, "web_search", {})
    assert res_web["status"] == "error"


# 7. Search Failure and Graceful Recovery Test
def test_search_failure_handling(sample_image_bytes, dummy_embedding, monkeypatch):
    state = ResearchState(input_image_bytes=sample_image_bytes, target_embedding=dummy_embedding)
    
    # Simulate missing SerpApi key
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    res = execute_web_search(state, {"query": "sample query", "reason": "testing"})
    assert res["status"] == "error"
    assert "Missing SERPAPI_KEY" in res["error"]


# 8. Page Image Extraction Test
def test_extract_page_images(sample_image_bytes, dummy_embedding, monkeypatch):
    state = ResearchState(input_image_bytes=sample_image_bytes, target_embedding=dummy_embedding)
    
    html_mock = """
    <html>
      <head>
        <meta property="og:image" content="https://cdn.example.com/social_banner.jpg" />
      </head>
      <body>
        <img src="/photos/speaker1.jpg" alt="Speaker 1" />
        <img src="https://cdn.example.com/attendee.png" />
        <img src="https://example.com/icon_1x1.png" /> <!-- Junk -->
        <a href="https://cdn.example.com/full_gallery.jpg">Gallery Photo</a>
      </body>
    </html>
    """
    
    class MockResponse:
        status_code = 200
        text = html_mock
        content = html_mock.encode("utf-8")
        
    monkeypatch.setattr("agent.tools._http_get", lambda url, timeout=10: MockResponse())
    
    res = execute_extract_page_images(state, {"url": "https://example.com/event", "reason": "Harvesting gallery"})
    assert res["status"] == "success"
    assert res["total_extracted"] >= 3
    # Check that candidates were added to state
    img_urls = [cand["image_url"] for cand in state.candidate_images]
    assert "https://cdn.example.com/social_banner.jpg" in img_urls
    assert "https://example.com/photos/speaker1.jpg" in img_urls


# 9. Candidate Face Matching with Real Fixtures Test
def test_face_match_with_fixtures(sample_image_bytes, sample_image_bytes_b):
    from faceid.encoder import encode_face
    emb_a = encode_face(sample_image_bytes)
    
    state = ResearchState(
        input_image_bytes=sample_image_bytes,
        target_embedding=emb_a,
        tolerance=0.35,
    )
    
    # Add fixture B to candidate images
    state.candidate_images.append({
        "image_url": "https://example.com/person1_b.jpg",
        "source_page": "https://example.com/post1",
        "_bytes": sample_image_bytes_b,
        "distance": None,
        "matched": False,
    })
    
    res = execute_face_match(state, {
        "image_url": "https://example.com/person1_b.jpg",
        "reason": "Verify candidate fixture B against input fixture A",
    })
    
    assert res["status"] == "evaluated"
    assert res["matched"] is True
    assert res["best_distance"] < 0.35
    assert len(state.verified_candidates) == 1


# 10. Agent Budget Enforcement Test
def test_agent_budget_enforcement(sample_image_bytes, dummy_embedding):
    state = ResearchState(
        input_image_bytes=sample_image_bytes,
        target_embedding=dummy_embedding,
        max_iterations=2,
        max_queries=2,
        max_pages=2,
    )
    
    assert state.is_budget_exhausted()[0] is False
    
    state.iteration = 2
    exhausted, reason = state.is_budget_exhausted()
    assert exhausted is True
    assert "Maximum iterations" in reason


# 11. Heuristic Planner Workflow Test
def test_heuristic_planner_decision_flow(sample_image_bytes, dummy_embedding):
    state = ResearchState(input_image_bytes=sample_image_bytes, target_embedding=dummy_embedding)
    planner = HeuristicAgentClient()
    
    # Action 1: Should decide OCR or image analysis
    act1 = planner.decide_next_action(state)
    assert act1.tool_name in ("extract_ocr", "analyze_image")
    
    # Mark OCR as run with keywords
    state.action_history.append({"tool": "extract_ocr", "arguments": {}})
    state.action_history.append({"tool": "analyze_image", "arguments": {}})
    state.ocr_results["keywords"] = ["DevCon", "AlphaCorp"]
    
    # Action 2: Should decide web search for OCR keywords
    act2 = planner.decide_next_action(state)
    assert act2.tool_name == "web_search"
    assert "DevCon" in act2.arguments.get("query", "") or "AlphaCorp" in act2.arguments.get("query", "")


# 12. Blockchain Record Integrity Test for Agent Output
def test_blockchain_anchoring_agent_output(tmp_path):
    chain_file = str(tmp_path / "test_chain.json")
    chain = LocalBlockchain(filepath=chain_file)
    
    post_record = {
        "platform": "x",
        "post_url": "https://x.com/lead_developer/status/9876543210",
        "author": "@lead_developer",
        "text": "Excited to share memories from the hackathon!",
        "posted_at": "2026-03-01T12:00:00Z",
        "image_sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    }
    
    canonical = make_canonical_record(post_record)
    content_hash = compute_record_hash(canonical)
    
    receipt = chain.anchor(content_hash=content_hash, post_url=post_record["post_url"])
    assert receipt["block_index"] == 1
    assert receipt["content_hash"] == content_hash
    
    is_valid, proof, _ = chain.verify(content_hash, expected_post_url=post_record["post_url"])
    assert is_valid is True
    assert proof["content_hash"] == content_hash


# 13. End-to-End Integration Test with Mocked Multi-Hop OSINT
def test_end_to_end_mocked_multihop_agent(sample_image_bytes, sample_image_bytes_b, tmp_path, monkeypatch):
    """
    Validates complete multi-hop pipeline:
    Input photo -> OCR clue -> Web search -> Discovered event page ->
    Extracted candidate image -> InsightFace verification -> Verified post -> Blockchain anchor.
    """
    from faceid.encoder import encode_face, extract_face_crop
    _, target_emb, face_meta = extract_face_crop(sample_image_bytes)
    
    # Mock OCR to return distinctive event keyword
    monkeypatch.setattr(
        "agent.tools.extract_image_text_and_keywords",
        lambda img: {"full_text": "Global Hacker Summit 2026", "keywords": ["Global Hacker Summit"], "segments": []}
    )
    
    # Mock Web Search to return event page lead
    def mock_web_search(state, args):
        q = args.get("query", "")
        state.mark_query_seen(q)
        lead = {
            "title": "Global Hacker Summit 2026 Attendee Gallery",
            "url": "https://summit.example.com/gallery",
            "link": "https://summit.example.com/gallery",
            "snippet": "Photos from day 1 networking event.",
        }
        state.search_results.append(lead)
        return {"status": "success", "results": [lead]}
        
    monkeypatch.setattr("agent.tools.execute_web_search", mock_web_search)
    
    # Mock Page Fetch to return HTML containing person1_b candidate photo
    html_page = """
    <html>
      <head><title>Attendee Photos</title></head>
      <body>
        <img src="https://summit.example.com/attendee_photo_b.jpg" alt="Attendee photo" />
      </body>
    </html>
    """
    
    class MockResp:
        status_code = 200
        text = html_page
        content = sample_image_bytes_b
        
    monkeypatch.setattr("agent.tools._http_get", lambda url, timeout=10: MockResp())
    
    # Mock fetch_post to return canonical post metadata
    monkeypatch.setattr(
        "agent.tools.fetch_post",
        lambda url: {
            "platform": "web",
            "post_url": "https://summit.example.com/gallery",
            "author": "Summit Organizer",
            "text": "Attendee Gallery 2026",
            "posted_at": "2026-03-01T15:00:00Z",
            "_image_bytes": sample_image_bytes_b,
        }
    )
    
    agent = ResearchAgent(max_iterations=8)
    result = agent.run(
        input_image_bytes=sample_image_bytes,
        target_embedding=target_emb,
        target_face_meta=face_meta,
        tolerance=0.35,
        offline_demo=False,
    )
    
    assert result.is_match_found is True
    assert result.accepted_record is not None
    assert result.accepted_distance < 0.35
    assert len(result.evidence_chain) >= 2
    
    # Verify anchoring of result
    chain_file = str(tmp_path / "chain.json")
    chain = LocalBlockchain(filepath=chain_file)
    canonical = make_canonical_record(result.accepted_record)
    content_hash = compute_record_hash(canonical)
    receipt = chain.anchor(content_hash, canonical["post_url"])
    assert receipt["block_index"] == 1
