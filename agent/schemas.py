"""Structured tool schemas for LLM function calling (Gemini & OpenAI compliant)."""

from __future__ import annotations
from typing import Any

AGENT_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "analyze_image",
        "description": "Inspects the input photograph to extract observable, searchable visual clues (scene description, clothing, visible text/signage, logos, potential organizations or event badges). Does NOT attempt to identify people by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "focus_area": {
                    "type": "string",
                    "description": "Area to analyze: 'full_scene', 'clothing_and_lanyard', 'background_signage', or 'logos'.",
                    "enum": ["full_scene", "clothing_and_lanyard", "background_signage", "logos"],
                },
                "reason": {
                    "type": "string",
                    "description": "Why this visual analysis is being conducted.",
                },
            },
            "required": ["focus_area", "reason"],
        },
    },
    {
        "name": "extract_ocr",
        "description": "Runs Optical Character Recognition (OCR) on the image or targeted crops (e.g. lanyards, name badges, event banners, credentials) to extract readable text phrases and potential keywords.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_region": {
                    "type": "string",
                    "description": "Specific region to OCR: 'full_image', 'chest_lanyard', 'signage', or 'badge'.",
                    "enum": ["full_image", "chest_lanyard", "signage", "badge"],
                },
                "reason": {
                    "type": "string",
                    "description": "Rationale for choosing this region for text extraction.",
                },
            },
            "required": ["target_region", "reason"],
        },
    },
    {
        "name": "reverse_image_search",
        "description": "Executes reverse image search via Google Lens or Yandex using either the focused facial portrait crop or the full situational scene.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_perspective": {
                    "type": "string",
                    "description": "Perspective to search: 'face_crop' (recommended for facial identity) or 'full_scene' (recommended for clothing/event background).",
                    "enum": ["face_crop", "full_scene"],
                },
                "engine": {
                    "type": "string",
                    "description": "Reverse image search engine: 'google_lens' or 'yandex'.",
                    "enum": ["google_lens", "yandex"],
                },
                "reason": {
                    "type": "string",
                    "description": "Why this perspective and engine were chosen.",
                },
            },
            "required": ["image_perspective", "engine", "reason"],
        },
    },
    {
        "name": "web_search",
        "description": "Performs a public web search query via Google/SerpApi. Used to search distinctive organization names, event titles, venues, or rare clue combinations discovered during the investigation.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string. Prioritize high-information clues (e.g. 'Organization Name' event photos) rather than generic descriptions.",
                },
                "domain_restriction": {
                    "type": "string",
                    "description": "Optional domain filter (e.g. 'x.com', 'linkedin.com', 'eventbrite.com'). Leave empty for general web.",
                },
                "reason": {
                    "type": "string",
                    "description": "Investigative hypothesis driving this specific search query.",
                },
            },
            "required": ["query", "reason"],
        },
    },
    {
        "name": "search_social_platform",
        "description": "Searches for public indexed posts on a specific platform (X/Twitter, LinkedIn, Instagram, Reddit, Facebook) using site-restricted queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "description": "Platform to search: 'x.com', 'linkedin.com', 'instagram.com', 'reddit.com', or 'facebook.com'.",
                    "enum": ["x.com", "linkedin.com", "instagram.com", "reddit.com", "facebook.com"],
                },
                "query": {
                    "type": "string",
                    "description": "Search query keywords (e.g. 'organization name' or '#eventtag photos').",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this platform and query are expected to yield candidate posts.",
                },
            },
            "required": ["platform", "query", "reason"],
        },
    },
    {
        "name": "open_url",
        "description": "Navigates to and parses a public webpage found in search results. Extracts page title, article text, author/account, published date, canonical URL, and OpenGraph metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The exact URL to open. Must come from real search results; never hallucinated.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this page is relevant to the investigation.",
                },
            },
            "required": ["url", "reason"],
        },
    },
    {
        "name": "extract_page_images",
        "description": "Extracts all candidate photo URLs from a public webpage. Turns event galleries, attendee lists, or social post pages into a local candidate image corpus for biometric verification.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The page URL from which to harvest public images.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this page is expected to contain photographs of the subject.",
                },
            },
            "required": ["url", "reason"],
        },
    },
    {
        "name": "download_candidate_image",
        "description": "Safely downloads a candidate image URL into the local research corpus and checks for perceptual duplicates (aHash).",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "Direct image URL to download.",
                },
                "source_page": {
                    "type": "string",
                    "description": "The webpage URL where this image was discovered.",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for selecting this candidate image.",
                },
            },
            "required": ["image_url", "source_page", "reason"],
        },
    },
    {
        "name": "face_match",
        "description": "Runs deterministic InsightFace biometric verification on a candidate image. Compares target face embedding against ALL detected faces in the image using cosine distance. Biometric threshold is non-negotiable (tol < 0.35).",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "Candidate image URL or downloaded image key to evaluate biometrically.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this image should be evaluated with InsightFace.",
                },
            },
            "required": ["image_url", "reason"],
        },
    },
    {
        "name": "inspect_candidate_post",
        "description": "Once a candidate image matches biometrically, resolves and constructs the full canonical social/web post record (URL, author, post text, timestamp, image SHA-256) for blockchain anchoring.",
        "parameters": {
            "type": "object",
            "properties": {
                "post_url": {
                    "type": "string",
                    "description": "URL of the matching post/webpage.",
                },
                "matched_image_url": {
                    "type": "string",
                    "description": "The image URL that verified with InsightFace.",
                },
                "reason": {
                    "type": "string",
                    "description": "Confirmation of verified biometric match on this post.",
                },
            },
            "required": ["post_url", "matched_image_url", "reason"],
        },
    },
    {
        "name": "add_discovered_entity",
        "description": "Registers a newly discovered entity (organization, event, venue, public figure, username) to the research state to serve as a new search pivot.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Category of entity.",
                    "enum": ["organization", "event", "location", "username", "website"],
                },
                "name": {
                    "type": "string",
                    "description": "Name or handle of the entity.",
                },
                "context": {
                    "type": "string",
                    "description": "Context or page where this entity was discovered.",
                },
            },
            "required": ["category", "name", "context"],
        },
    },
    {
        "name": "finish_investigation",
        "description": "Stops the investigation loop when a satisfactory genuine matching post has been biometrically verified and fully resolved, or when the agent has exhausted all promising leads.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "'match_verified' if a confirmed candidate was found; 'no_match_found' if budget exhausted.",
                    "enum": ["match_verified", "no_match_found"],
                },
                "summary": {
                    "type": "string",
                    "description": "Concise summary of the findings and evidence chain.",
                },
            },
            "required": ["status", "summary"],
        },
    },
]


def get_gemini_tools_declaration() -> list[dict[str, Any]]:
    """Transforms AGENT_TOOLS_SCHEMA into Google Gemini FunctionDeclaration format."""
    function_declarations = []
    for tool in AGENT_TOOLS_SCHEMA:
        function_declarations.append({
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        })
    return [{"function_declarations": function_declarations}]


def get_openai_tools_declaration() -> list[dict[str, Any]]:
    """Transforms AGENT_TOOLS_SCHEMA into OpenAI standard tools format."""
    return [{"type": "function", "function": tool} for tool in AGENT_TOOLS_SCHEMA]
