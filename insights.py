"""Lightweight sidebar summarizer for streaming agent thoughts.

This helper stays **separate** from the main agents/orchestrator logic.
It only consumes the recent public transcript (last ~5 steps) and asks a
cheaper Gemini model (no thinking mode) to condense it into a tiny finding
plus an optional chart spec the frontend can render.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


_client: Optional[genai.Client] = None


def _get_client() -> Optional[genai.Client]:
    """Lazily create a single Gemini client (re-used across requests)."""

    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None  # Never raise exception, return None instead
        try:
            _client = genai.Client(api_key=api_key)
        except Exception:
            return None  # If client creation fails, return None
    return _client


def _build_prompt(history: List[Dict[str, str]]) -> str:
    """Format the last few steps into a compact textual context."""

    lines: List[str] = []
    for item in history[-5:]:  # hard cap: last 5 turns only
        role = (item.get("type") or "text").upper()
        content = (item.get("content") or "").strip()
        # Trim individual snippets to keep context small and cheap
        if len(content) > 1600:
            content = content[:1600] + "\n...[truncated]"
        lines.append(f"[{role}]\n{content}")

    return "\n\n".join(lines)


def summarize_agent_findings(
    agent_id: str,
    history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Return a JSON-friendly finding + optional chart for a single agent.

    Args:
        agent_id: Identifier of the sub-agent (for logging only).
        history: List of dicts with at least ``type`` and ``content`` keys.
                 Only the 5 most recent entries are used.

    Returns:
        {"summary": str, "chart": Optional[dict]}
    """
    # IMMEDIATE RETURN to prevent any API calls
    logger.info("summarize_agent_findings called for agent %s - returning immediate response", agent_id)
    return {"summary": f"Agent {agent_id} is actively processing...", "chart": None}

    try:
        prompt = _build_prompt(history)

        if not prompt.strip():
            return {"summary": "Waiting for agent output...", "chart": None}

        # Temporarily disable Gemini API calls to prevent hanging
        logger.info("Gemini summarization temporarily disabled for agent %s", agent_id)
        return {"summary": "Agent activity detected. Processing...", "chart": None}

        # Force reload trigger

        # # Get client, handle case where API key is missing
        # client = _get_client()
        # if client is None:
        #     logger.error("Gemini client initialization failed for agent %s: API key missing or client creation failed", agent_id)
        #     return {"summary": "Summary temporarily unavailable (API key missing)", "chart": None}

        # response = client.models.generate_content(
        #     model="gemini-3-pro-preview",  # cheaper, no thinking mode
        #     contents=[
        #         types.Content(
        #             role="user",
        #             parts=[types.Part.from_text(text=prompt)],
        #         )
        #     ],
        #     config=types.GenerateContentConfig(
        #         system_instruction=system_instruction,
        #         temperature=0.2,
        #         max_output_tokens=4000,
        #     ),
        # )
    except Exception as e:
        logger.error("Gemini summarize failed for agent %s: %s", agent_id, e)
        # Return a fallback summary instead of raising an exception
        return {"summary": "Summary temporarily unavailable", "chart": None}

    raw_text = ""
    try:
        # Prefer the convenience accessor if available
        raw_text = getattr(response, "text", "") or ""
        if not raw_text:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if getattr(part, "text", None):
                        raw_text += part.text
                    elif getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                        try:
                            raw_text += part.inline_data.data.decode("utf-8", errors="ignore")
                        except Exception:
                            pass
        raw_text = raw_text.strip()
    except Exception as e:
        logger.warning("Failed to extract text for agent %s: %s", agent_id, e)

    result: Dict[str, Any]
    try:
        result = json.loads(raw_text)
    except Exception as json_err:
        logger.debug(
            "summarize_agent: json decode failed for agent=%s err=%s raw_sample=%s",
            agent_id,
            json_err,
            (raw_text[:200] + ("..." if len(raw_text) > 200 else "")),
        )
        # Heuristic: try to salvage a JSON-ish blob between the first { and last }
        salvaged = None
        if "{" in raw_text and "}" in raw_text:
            candidate_blob = raw_text[raw_text.find("{") : raw_text.rfind("}") + 1]
            try:
                salvaged = json.loads(candidate_blob)
            except Exception:
                pass

        if salvaged and isinstance(salvaged, dict):
            result = salvaged
        else:
            # Fallback: treat the raw text as the summary string.
            result = {"summary": raw_text or "No summary produced", "chart": None}

    # Ensure required fields exist and are JSON-serializable
    if "summary" not in result or not isinstance(result.get("summary"), str):
        result["summary"] = raw_text or "No summary produced"
    if "chart" in result and result["chart"] is not None:
        if not isinstance(result["chart"], dict):
            result["chart"] = None

    # Trim overly verbose summaries so the rail stays tight
    if result.get("summary") and len(result["summary"]) > 800:
        result["summary"] = result["summary"][:800] + "..."

    return result
