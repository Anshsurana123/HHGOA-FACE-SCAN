"""Persistent in-memory research state and evidence graph for OSINT discovery."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import numpy as np


class EvidenceNode:
    """Represents a discrete discovery or clue node in the OSINT investigation."""

    def __init__(
        self,
        node_id: str,
        node_type: str,
        label: str,
        metadata: Optional[dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ):
        self.node_id = node_id
        self.node_type = node_type  # 'image', 'ocr', 'clue', 'query', 'url', 'entity', 'candidate_image', 'match'
        self.label = label
        self.metadata = metadata or {}
        self.parent_id = parent_id
        self.timestamp = time.time()
        self.timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))

    def to_dict(self) -> dict[str, Any]:
        clean_meta = {}
        for k, v in self.metadata.items():
            if isinstance(v, (bytes, bytearray)):
                clean_meta[k] = f"<{len(v)} bytes>"
            elif isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        return {
            "id": self.node_id,
            "type": self.node_type,
            "label": self.label,
            "metadata": clean_meta,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp_iso,
        }



class EvidenceEdge:
    """Represents a causal or navigational relationship between evidence nodes."""

    def __init__(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        reason: str = "",
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.relation = relation  # 'extracted_from', 'queried_for', 'discovered_page', 'contained_image', 'biometrically_verified'
        self.reason = reason
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation,
            "reason": self.reason,
        }


class EvidenceGraph:
    """Directed graph capturing why and how every piece of evidence was found."""

    def __init__(self):
        self.nodes: dict[str, EvidenceNode] = {}
        self.edges: list[EvidenceEdge] = []
        self._counter = 0

    def add_node(
        self,
        node_type: str,
        label: str,
        metadata: Optional[dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        relation: str = "",
        reason: str = "",
    ) -> EvidenceNode:
        self._counter += 1
        node_id = f"n{self._counter}_{node_type}"
        node = EvidenceNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            metadata=metadata,
            parent_id=parent_id,
        )
        self.nodes[node_id] = node

        if parent_id and parent_id in self.nodes:
            self.edges.append(EvidenceEdge(
                source_id=parent_id,
                target_id=node_id,
                relation=relation or "leads_to",
                reason=reason,
            ))
        return node

    def add_edge(self, source_id: str, target_id: str, relation: str, reason: str = ""):
        if source_id in self.nodes and target_id in self.nodes:
            self.edges.append(EvidenceEdge(source_id, target_id, relation, reason))

    def get_chain_to_node(self, node_id: str) -> list[dict[str, Any]]:
        """Backtracks from a given node to the root node to produce a linear evidence chain."""
        chain = []
        curr_id = node_id
        visited = set()
        while curr_id and curr_id not in visited and curr_id in self.nodes:
            visited.add(curr_id)
            node = self.nodes[curr_id]
            chain.append(node.to_dict())
            curr_id = node.parent_id
        chain.reverse()
        return chain

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
        }


def normalize_search_query(query: str) -> str:
    """Normalizes query text for deduplication (case, extra spaces, punctuation)."""
    q = query.strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


def normalize_target_url(url: str) -> str:
    """Normalizes URL by stripping trailing slashes, fragments, and tracking parameters."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        # Remove common tracking parameters
        clean_params = []
        for k, v in parse_qsl(parsed.query):
            if not k.startswith("utm_") and k not in ("fbclid", "igshid", "ref_src", "s", "t"):
                clean_params.append((k, v))
        path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
        return urlunparse((
            parsed.scheme.lower() or "https",
            netloc,
            path,
            parsed.params,
            urlencode(clean_params),
            "",
        ))
    except Exception:
        return url.strip().rstrip("/")


class ResearchState:
    """In-memory research state object for an OSINT agent investigation run."""

    def __init__(
        self,
        input_image_bytes: bytes,
        input_image_path: Optional[str] = None,
        target_embedding: Optional[np.ndarray] = None,
        target_face_meta: Optional[dict[str, Any]] = None,
        tolerance: float = 0.35,
        max_iterations: Optional[int] = None,
        max_queries: Optional[int] = None,
        max_pages: Optional[int] = None,
        max_images: Optional[int] = None,
    ):
        self.input_image_bytes = input_image_bytes
        self.input_image_path = input_image_path
        self.target_embedding = target_embedding
        self.target_face_meta = target_face_meta or {}
        self.tolerance = tolerance

        # Configurable investigation budgets
        self.max_iterations = max_iterations or int(os.getenv("MAX_AGENT_ITERATIONS", "15"))
        self.max_queries = max_queries or int(os.getenv("MAX_SEARCH_QUERIES", "30"))
        self.max_pages = max_pages or int(os.getenv("MAX_PAGES_TO_OPEN", "30"))
        self.max_candidate_images = max_images or int(os.getenv("MAX_CANDIDATE_IMAGES", "150"))
        self.max_download_bytes = int(os.getenv("MAX_DOWNLOAD_BYTES_PER_IMAGE", "10485760"))  # 10MB
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "10"))

        # Counters
        self.iteration = 0
        self.total_queries = 0
        self.total_pages_opened = 0
        self.total_images_extracted = 0
        self.total_face_matches = 0

        # State tracking sets for deduplication
        self.normalized_queries: set[str] = set()
        self.visited_urls: set[str] = set()
        self.rejected_urls: set[str] = set()
        self.downloaded_image_urls: set[str] = set()
        self.failed_queries: set[str] = set()

        # Evidence & Clues
        self.ocr_results: dict[str, Any] = {"full_text": "", "segments": [], "keywords": []}
        self.visual_clues: dict[str, Any] = {
            "scene_description": "",
            "objects": [],
            "clothing_clues": [],
            "visible_signage": [],
            "logos": [],
            "possible_organizations": [],
            "possible_locations": [],
            "possible_events": [],
            "other_searchable_clues": [],
        }
        self.discovered_entities: dict[str, set[str]] = {
            "organizations": set(),
            "locations": set(),
            "events": set(),
            "usernames": set(),
            "websites": set(),
        }

        # Knowledge & Search Graph
        self.evidence_graph = EvidenceGraph()
        self.root_node = self.evidence_graph.add_node(
            node_type="image",
            label="Input Face Photo",
            metadata={"size_bytes": len(input_image_bytes), "path": input_image_path},
        )

        # Discovered Leads & Candidates
        self.search_results: list[dict[str, Any]] = []
        self.candidate_pages: list[dict[str, Any]] = []
        self.candidate_images: list[dict[str, Any]] = []  # {image_url, source_page, ahash, distance, matched}
        self.verified_candidates: list[dict[str, Any]] = []
        self.best_candidate: Optional[dict[str, Any]] = None

        # Execution Trail & Telemetry
        self.action_history: list[dict[str, Any]] = []
        self.telemetry_logs: list[str] = []
        self.start_time = time.time()
        self.completed = False
        self.termination_reason = ""

    def log(self, message: str):
        t_str = time.strftime("%H:%M:%S")
        entry = f"[{t_str}] {message}"
        self.telemetry_logs.append(entry)

    def is_query_seen(self, query: str) -> bool:
        norm = normalize_search_query(query)
        return norm in self.normalized_queries

    def mark_query_seen(self, query: str):
        norm = normalize_search_query(query)
        self.normalized_queries.add(norm)
        self.total_queries += 1

    def is_url_visited(self, url: str) -> bool:
        norm = normalize_target_url(url)
        return norm in self.visited_urls

    def mark_url_visited(self, url: str):
        norm = normalize_target_url(url)
        self.visited_urls.add(norm)
        self.total_pages_opened += 1

    def is_image_seen(self, image_url: str) -> bool:
        norm = normalize_target_url(image_url)
        return norm in self.downloaded_image_urls

    def mark_image_seen(self, image_url: str):
        norm = normalize_target_url(image_url)
        self.downloaded_image_urls.add(norm)

    def add_entity(self, category: str, value: str, parent_node_id: Optional[str] = None):
        if not value or not value.strip():
            return
        val_clean = value.strip()
        cat_key = category.lower()
        if cat_key in self.discovered_entities:
            if val_clean not in self.discovered_entities[cat_key]:
                self.discovered_entities[cat_key].add(val_clean)
                self.evidence_graph.add_node(
                    node_type="entity",
                    label=f"[{category.title()}] {val_clean}",
                    metadata={"category": cat_key, "value": val_clean},
                    parent_id=parent_node_id,
                    relation="identified_entity",
                    reason=f"Extracted {category} clue from investigation",
                )
                self.log(f"Discovered {category}: {val_clean}")

    def is_budget_exhausted(self) -> tuple[bool, str]:
        """Checks whether the agent has reached any configured budget boundaries."""
        if self.iteration >= self.max_iterations:
            return True, f"Maximum iterations reached ({self.max_iterations})"
        if self.total_queries >= self.max_queries:
            return True, f"Search query budget exhausted ({self.max_queries} queries)"
        if self.total_pages_opened >= self.max_pages:
            return True, f"Page exploration budget exhausted ({self.max_pages} pages)"
        if len(self.candidate_images) >= self.max_candidate_images:
            return True, f"Candidate image limit reached ({self.max_candidate_images} images)"
        return False, ""

    def get_summary_dict(self) -> dict[str, Any]:
        """Returns structured JSON summary of the research state for UI and persistence."""
        return {
            "iterations": self.iteration,
            "elapsed_seconds": round(time.time() - self.start_time, 2),
            "total_queries": self.total_queries,
            "total_pages_opened": self.total_pages_opened,
            "total_images_collected": len(self.candidate_images),
            "total_faces_evaluated": self.total_face_matches,
            "verified_candidates_count": len(self.verified_candidates),
            "discovered_entities": {k: list(v) for k, v in self.discovered_entities.items()},
            "queries_executed": list(self.normalized_queries),
            "visited_urls": list(self.visited_urls),
            "completed": self.completed,
            "termination_reason": self.termination_reason,
            "evidence_graph": self.evidence_graph.to_dict(),
        }
