"""LLM Provider abstraction layer supporting Gemini, OpenAI, and Heuristic Offline clients."""

from __future__ import annotations

import json
import os
from typing import Any, Optional
import requests

from agent.state import ResearchState
from agent.schemas import AGENT_TOOLS_SCHEMA, get_gemini_tools_declaration, get_openai_tools_declaration
from agent.prompts import OSINT_RESEARCHER_SYSTEM_PROMPT, build_agent_observation_prompt


class ToolCall:
    """Represents a validated tool call selected by the agent planner."""

    def __init__(self, tool_name: str, arguments: dict[str, Any], raw_response: str = ""):
        self.tool_name = tool_name
        self.arguments = arguments
        self.raw_response = raw_response

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool_name,
            "arguments": self.arguments,
        }


class LLMClient:
    """Abstract base class for LLM research agents."""

    def decide_next_action(self, state: ResearchState) -> ToolCall:
        raise NotImplementedError


class GeminiLLMClient(LLMClient):
    """Native Google Gemini client using structured function calling."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gemini-2.5-flash")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be provided.")

    def decide_next_action(self, state: ResearchState) -> ToolCall:
        prompt_text = build_agent_observation_prompt(state)
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "system_instruction": {
                "parts": [{"text": OSINT_RESEARCHER_SYSTEM_PROMPT}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt_text}],
                }
            ],
            "tools": get_gemini_tools_declaration(),
            "tool_config": {
                "function_calling_config": {
                    "mode": "ANY"  # Force model to choose a tool call
                }
            },
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 800,
            }
        }

        resp = requests.post(endpoint, json=payload, timeout=25)
        if resp.status_code != 200:
            state.log(f"[LLM] Gemini API error ({resp.status_code}): {resp.text[:200]}")
            # Fallback to heuristic selection if API error occurs
            return HeuristicAgentClient().decide_next_action(state)

        data = resp.json()
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return HeuristicAgentClient().decide_next_action(state)

            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "functionCall" in part:
                    fn_name = part["functionCall"]["name"]
                    fn_args = part["functionCall"].get("args", {})
                    return ToolCall(tool_name=fn_name, arguments=fn_args, raw_response=json.dumps(part))
        except Exception as exc:
            state.log(f"[LLM] Failed to parse Gemini tool call: {exc}")

        return HeuristicAgentClient().decide_next_action(state)


class OpenAILLMClient(LLMClient):
    """OpenAI-compatible client with tool_calls schema."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be provided.")

    def decide_next_action(self, state: ResearchState) -> ToolCall:
        prompt_text = build_agent_observation_prompt(state)
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": OSINT_RESEARCHER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "tools": get_openai_tools_declaration(),
            "tool_choice": "required",
            "temperature": 0.2,
        }

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=25)
        if resp.status_code != 200:
            state.log(f"[LLM] OpenAI API error: {resp.text[:200]}")
            return HeuristicAgentClient().decide_next_action(state)

        data = resp.json()
        try:
            choice = data["choices"][0]
            tool_calls = choice["message"].get("tool_calls", [])
            if tool_calls:
                fn = tool_calls[0]["function"]
                fn_name = fn["name"]
                fn_args = json.loads(fn.get("arguments", "{}"))
                return ToolCall(tool_name=fn_name, arguments=fn_args, raw_response=json.dumps(tool_calls[0]))
        except Exception as exc:
            state.log(f"[LLM] Failed to parse OpenAI tool call: {exc}")

        return HeuristicAgentClient().decide_next_action(state)


class HeuristicAgentClient(LLMClient):
    """
    Deterministic OSINT heuristic planner.
    Operates when no external LLM API key is configured or during offline testing/fallback.
    Executes intelligent multi-hop evidence traversal:
    1. Image Analysis & OCR
    2. High-Entropy OCR & Discovered Entity search queries
    3. Promising page opening & image corpus harvesting
    4. Deterministic InsightFace matching
    5. Termination upon verified match
    """

    def decide_next_action(self, state: ResearchState) -> ToolCall:
        # Check termination condition: Has any candidate verified biometrically?
        for cand in state.candidate_images:
            if cand.get("matched") and cand.get("distance", 1.0) < state.tolerance:
                source_page = cand.get("source_page", "")
                img_url = cand.get("image_url", "")
                if state.best_candidate and state.best_candidate.get("post_url"):
                    return ToolCall(
                        tool_name="finish_investigation",
                        arguments={
                            "status": "match_verified",
                            "summary": f"Biometrically verified matching post resolved at {state.best_candidate['post_url']} with cosine distance {cand.get('distance')}.",
                        },
                    )
                elif source_page:
                    return ToolCall(
                        tool_name="inspect_candidate_post",
                        arguments={
                            "post_url": source_page,
                            "matched_image_url": img_url,
                            "reason": "Biometric match confirmed. Resolving full canonical post metadata.",
                        },
                    )

        # 1. First iteration: Run OCR and visual analysis
        if not state.ocr_results.get("full_text") and "extract_ocr" not in [a["tool"] for a in state.action_history]:
            return ToolCall(
                tool_name="extract_ocr",
                arguments={
                    "target_region": "full_image",
                    "reason": "Initial text extraction to discover readable organization/event clues.",
                },
            )

        if not state.visual_clues.get("scene_description") and "analyze_image" not in [a["tool"] for a in state.action_history]:
            return ToolCall(
                tool_name="analyze_image",
                arguments={
                    "focus_area": "full_scene",
                    "reason": "Analyze setting, attire, and visible signage.",
                },
            )

        # 2. If OCR keywords exist, synthesize high-entropy targeted search queries
        keywords = state.ocr_results.get("keywords", [])
        if keywords:
            # Try specific proper noun keywords
            for kw in keywords:
                q1 = f'"{kw}"'
                if not state.is_query_seen(q1):
                    return ToolCall(
                        tool_name="web_search",
                        arguments={"query": q1, "reason": f"Search distinctive OCR entity {kw}"},
                    )
                q2 = f'site:x.com "{kw}"'
                if not state.is_query_seen(q2):
                    return ToolCall(
                        tool_name="web_search",
                        arguments={"query": q2, "reason": f"Search X/Twitter for OCR entity {kw}"},
                    )

        # 3. If unevaluated candidate images exist, run face_match on them!
        unevaluated_images = [img for img in state.candidate_images if img.get("distance") is None]
        if unevaluated_images:
            next_img = unevaluated_images[0]
            return ToolCall(
                tool_name="face_match",
                arguments={
                    "image_url": next_img["image_url"],
                    "reason": "Biometrically evaluate discovered candidate photograph with InsightFace.",
                },
            )

        # 4. If unexplored search results exist, open the most promising page and harvest its images
        for res in state.search_results:
            link = res.get("url") or res.get("link") or ""
            if link and not state.is_url_visited(link) and link not in state.rejected_urls:
                return ToolCall(
                    tool_name="open_url",
                    arguments={
                        "url": link,
                        "reason": f"Inspect promising search lead ({res.get('title', '')[:40]}) to harvest candidate gallery.",
                    },
                )

        # 5. Reverse Image Search (Probe facial crop if not yet executed)
        rev_searches = [a for a in state.action_history if a["tool"] == "reverse_image_search"]
        if not rev_searches and os.getenv("SERPAPI_KEY"):
            return ToolCall(
                tool_name="reverse_image_search",
                arguments={
                    "image_perspective": "face_crop",
                    "engine": "google_lens",
                    "reason": "Query Google Lens with focused biometric face portrait probe.",
                },
            )

        # 6. Fallback: Search discovered entities
        for cat, entities in state.discovered_entities.items():
            for ent in entities:
                q = f'"{ent}" photos'
                if not state.is_query_seen(q):
                    return ToolCall(
                        tool_name="web_search",
                        arguments={"query": q, "reason": f"Search for public photo galleries related to {ent}."},
                    )

        # Budget / leads exhausted
        return ToolCall(
            tool_name="finish_investigation",
            arguments={
                "status": "no_match_found",
                "summary": "Exhausted all available investigative search leads and candidate images.",
            },
        )


def get_llm_client() -> LLMClient:
    """Factory creating configured LLM client with automatic fallback."""
    provider = os.getenv("LLM_PROVIDER", "").lower()
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if provider == "gemini" or (not provider and gemini_key):
        try:
            return GeminiLLMClient()
        except Exception:
            return HeuristicAgentClient()

    if provider == "openai" or (not provider and openai_key):
        try:
            return OpenAILLMClient()
        except Exception:
            return HeuristicAgentClient()

    # Default to deterministic heuristic planner
    return HeuristicAgentClient()
