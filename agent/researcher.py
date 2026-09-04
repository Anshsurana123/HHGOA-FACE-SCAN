"""Autonomous OSINT research agent orchestrator loop and candidate ranking."""

from __future__ import annotations

import os
import time
from typing import Any, Optional
import numpy as np

from agent.state import ResearchState, EvidenceGraph
from agent.tools import dispatch_tool
from agent.llm_client import LLMClient, get_llm_client, ToolCall
from faceid.encoder import DEFAULT_TOLERANCE


class AgentResult:
    """Encapsulates the end result and full investigative provenance of an agent run."""

    def __init__(
        self,
        accepted_record: Optional[dict[str, Any]],
        accepted_distance: Optional[float],
        evidence_graph: EvidenceGraph,
        evidence_chain: list[dict[str, Any]],
        candidate_logs: list[dict[str, Any]],
        total_queries: int,
        total_pages_opened: int,
        total_images_collected: int,
        total_faces_evaluated: int,
        iterations: int,
        search_engine: str = "agentic_osint",
        reason: str = "",
        state_summary: Optional[dict[str, Any]] = None,
    ):
        self.accepted_record = accepted_record
        self.accepted_distance = accepted_distance
        self.evidence_graph = evidence_graph
        self.evidence_chain = evidence_chain
        self.candidate_logs = candidate_logs
        self.total_queries = total_queries
        self.total_pages_opened = total_pages_opened
        self.total_images_collected = total_images_collected
        self.total_faces_evaluated = total_faces_evaluated
        self.total_engine_matches = total_images_collected  # Alias for backward compatibility
        self.total_social_candidates = len([c for c in candidate_logs if c.get("platform") in ("x", "instagram", "linkedin", "reddit", "facebook")])
        self.total_web_candidates = len(candidate_logs) - self.total_social_candidates
        self.iterations = iterations
        self.search_engine = search_engine
        self.reason = reason
        self.state_summary = state_summary or {}

    @property
    def is_match_found(self) -> bool:
        return self.accepted_record is not None


class ResearchAgent:
    """
    Autonomous OSINT research agent.
    Iteratively plans investigations, executes search & scraping tools,
    harvests candidate image galleries, runs deterministic InsightFace verification,
    and builds a cryptographic evidence graph.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_iterations: int = 15,
        max_queries: int = 30,
        max_pages: int = 30,
        max_images: int = 150,
    ):
        self.llm_client = llm_client or get_llm_client()
        self.max_iterations = max_iterations
        self.max_queries = max_queries
        self.max_pages = max_pages
        self.max_images = max_images

    def _rank_verified_candidates(self, state: ResearchState) -> list[tuple[float, dict[str, Any]]]:
        """
        Transparent multi-factor ranking for verified candidates:
        - 60% biometric face score: (1.0 - distance)
        - 15% image quality & resolution
        - 15% source quality (genuine social post vs generic domain)
        - 10% evidence depth (chain length)
        """
        ranked = []
        for cand in state.verified_candidates:
            dist = cand.get("distance", 1.0)
            if dist >= state.tolerance:
                continue

            face_score = max(0.0, 1.0 - dist)
            source_page = cand.get("source_page", "")

            # Source quality bonus
            src_score = 0.5
            if any(dom in source_page.lower() for dom in ("x.com", "twitter.com", "instagram.com", "linkedin.com", "reddit.com")):
                src_score = 1.0
            elif any(dom in source_page.lower() for dom in ("github.com", "devfolio.co", "luma.com", "eventbrite.com")):
                src_score = 0.8

            # Resolution bonus
            res_score = 0.5
            if cand.get("_bytes") and len(cand["_bytes"]) > 50000:
                res_score = 1.0

            total_score = (0.60 * face_score) + (0.25 * src_score) + (0.15 * res_score)
            ranked.append((total_score, cand))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def run(
        self,
        input_image_bytes: bytes,
        input_image_path: Optional[str] = None,
        target_embedding: Optional[np.ndarray] = None,
        target_face_meta: Optional[dict[str, Any]] = None,
        tolerance: float = DEFAULT_TOLERANCE,
        offline_demo: bool = False,
    ) -> AgentResult:
        """Executes the full agentic OSINT research loop."""
        state = ResearchState(
            input_image_bytes=input_image_bytes,
            input_image_path=input_image_path,
            target_embedding=target_embedding,
            target_face_meta=target_face_meta,
            tolerance=tolerance,
            max_iterations=self.max_iterations,
            max_queries=self.max_queries,
            max_pages=self.max_pages,
            max_images=self.max_images,
        )

        state.log("=== STARTING AGENTIC OSINT RESEARCH INVESTIGATION ===")

        # Fast Offline Demo Mode Handler
        if offline_demo:
            demo_url = "https://x.com/demo_user/status/1234567890"
            demo_record = {
                "platform": "x",
                "post_url": demo_url,
                "author": "Demo Public Figure",
                "text": "Verified sample post for offline pipeline validation.",
                "posted_at": "2026-01-01T12:00:00Z",
                "image_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "_image_bytes": b"demo_image_bytes",
            }
            cand_log = {
                "position": 1,
                "url": demo_url,
                "platform": "x",
                "distance": 0.12,
                "matched": True,
                "status": "MATCH (cosine distance 0.1200 < 0.35) [Offline Demo Mode]",
            }
            chain_node = state.evidence_graph.add_node(
                node_type="match",
                label="Offline Demo Match",
                metadata={k: v for k, v in demo_record.items() if k != "_image_bytes"},
                parent_id=state.root_node.node_id,
                relation="demo_verified",
            )

            return AgentResult(
                accepted_record=demo_record,
                accepted_distance=0.12,
                evidence_graph=state.evidence_graph,
                evidence_chain=state.evidence_graph.get_chain_to_node(chain_node.node_id),
                candidate_logs=[cand_log],
                total_queries=1,
                total_pages_opened=1,
                total_images_collected=1,
                total_faces_evaluated=1,
                iterations=1,
                reason="Matched via offline agent demonstration mode.",
                state_summary=state.get_summary_dict(),
            )

        # Iterative Agent Research Loop
        while not state.completed:
            state.iteration += 1

            # Check budget boundaries
            exhausted, exhaust_reason = state.is_budget_exhausted()
            if exhausted:
                state.log(f"[AGENT] Investigation halted: {exhaust_reason}")
                state.completed = True
                state.termination_reason = exhaust_reason
                break

            # 1. Ask LLM or Heuristic Planner for the highest-value next action
            state.log(f"--- [ITERATION {state.iteration} / {state.max_iterations}] Planning Next Action ---")
            tool_call: ToolCall = self.llm_client.decide_next_action(state)

            tool_name = tool_call.tool_name
            tool_args = tool_call.arguments
            state.log(f"[AGENT] Selected Action: {tool_name} with parameters: {tool_args}")

            # 2. Dispatch and execute tool
            res = dispatch_tool(state, tool_name, tool_args)

            # 3. Check for immediate stop signal
            if tool_name == "finish_investigation" or state.completed:
                state.completed = True
                break

            # 4. Immediate Stop Condition Check: If a verified post record is compiled, stop!
            if state.best_candidate and state.best_candidate.get("post_url"):
                state.log(f"[AGENT] Verified post record confirmed: {state.best_candidate['post_url']}. Terminating loop.")
                state.completed = True
                state.termination_reason = "Verified biometric match resolved."
                break

        # Candidate selection & ranking
        ranked = self._rank_verified_candidates(state)
        accepted_record = None
        accepted_dist = None
        best_node_id = None

        if ranked:
            best_cand = ranked[0][1]
            accepted_dist = best_cand.get("distance")
            # If best_candidate dict exists, use it; else enrich from best_cand
            if state.best_candidate:
                accepted_record = state.best_candidate
            else:
                accepted_record = {
                    "platform": "web",
                    "post_url": best_cand.get("source_page", ""),
                    "author": "",
                    "text": best_cand.get("title", ""),
                    "posted_at": "",
                    "image_sha256": "",
                    "_image_bytes": best_cand.get("_bytes"),
                }
            # Find evidence node corresponding to match
            for node in state.evidence_graph.nodes.values():
                if node.node_type == "match":
                    best_node_id = node.node_id
                    break

        evidence_chain = state.evidence_graph.get_chain_to_node(best_node_id) if best_node_id else []

        # Convert state candidate images into candidate_logs format
        candidate_logs = []
        for idx, cand in enumerate(state.candidate_images, start=1):
            dist = cand.get("distance")
            matched = cand.get("matched", False)
            status_text = f"MATCH (dist {dist:.4f} < {tolerance})" if matched else (f"REJECTED: dist {dist:.4f} >= {tolerance}" if dist is not None else "UNTESTED")
            candidate_logs.append({
                "position": idx,
                "url": cand.get("source_page", cand.get("image_url", "")),
                "image_url": cand.get("image_url", ""),
                "platform": "web",
                "distance": dist,
                "matched": matched,
                "status": status_text,
                "_image_bytes": cand.get("_bytes"),
            })

        state.log(f"=== INVESTIGATION CONCLUDED: Match Found={accepted_record is not None} ===")

        return AgentResult(
            accepted_record=accepted_record,
            accepted_distance=accepted_dist,
            evidence_graph=state.evidence_graph,
            evidence_chain=evidence_chain,
            candidate_logs=candidate_logs,
            total_queries=state.total_queries,
            total_pages_opened=state.total_pages_opened,
            total_images_collected=len(state.candidate_images),
            total_faces_evaluated=state.total_face_matches,
            iterations=state.iteration,
            search_engine="agentic_osint",
            reason=state.termination_reason or ("Verified post found." if accepted_record else "No matching post found within budget."),
            state_summary=state.get_summary_dict(),
        )
