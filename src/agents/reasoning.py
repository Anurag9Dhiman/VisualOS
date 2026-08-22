"""Reasoning Agent — synthesises Vision + Memory into a directed Search brief."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from google import genai
from google.genai import types

from src import rate_limiter
from src.contracts import CostEntry, MemoryResult, ReasoningTrace, VisionResult
from src.cost_logger import log_cost
from src.prompts import REASONING_SYSTEM_PROMPT, GeoPoint, build_reasoning_user_message

logger = logging.getLogger("lens.reasoning")

_MODEL = "gemini-3.5-flash"
_TIMEOUT_S = 12.0


def _vision_dict(vision: VisionResult) -> dict:
    return {
        "entity_name": vision.entity_name,
        "entity_type": vision.entity_type,
        "confidence_level": vision.confidence_level,
        "evidence": vision.evidence,
        "alternatives": vision.alternatives,
        "failure_modes_checked": vision.failure_modes_checked,
        "needs_fallback": vision.needs_fallback,
    }


def _memory_dict(memory: MemoryResult | None) -> dict:
    if memory is None:
        return {"hits": [], "user_interests_snapshot": {}}
    return {
        "hits": [
            {
                "subject_name": h.subject_name,
                "summary": h.summary,
                "similarity_score": h.similarity_score,
            }
            for h in memory.hits
        ],
        "user_interests_snapshot": memory.user_interests_snapshot,
    }


async def run_reasoning_agent(
    vision: VisionResult | None,
    memory: MemoryResult | None,
    lat: float | None,
    lng: float | None,
    cost_log: list[CostEntry],
) -> ReasoningTrace:
    if vision is None:
        return ReasoningTrace(
            what_we_know="Vision did not return a result.",
            key_unknowns=["What entity is this?"],
            research_brief="Vision failed — attempt broad identification from location alone.",
            suggested_tool_priority=[],
            memory_context="",
        )

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    location = GeoPoint(lat=lat or 0.0, lng=lng or 0.0)
    user_msg = build_reasoning_user_message(_vision_dict(vision), _memory_dict(memory), location)

    await rate_limiter.acquire(_MODEL)
    resp = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=_MODEL,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=REASONING_SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=600,
            ),
        ),
        timeout=_TIMEOUT_S,
    )

    usage = resp.usage_metadata
    cost_log.append(
        log_cost(
            "reasoning",
            _MODEL,
            (usage.prompt_token_count or 0) if usage else 0,
            (usage.candidates_token_count or 0) if usage else 0,
        )
    )

    try:
        data = json.loads(resp.text or "{}")
    except json.JSONDecodeError:
        logger.warning("Reasoning agent returned invalid JSON — using defaults")
        data = {}

    return ReasoningTrace(
        what_we_know=data.get("what_we_know", ""),
        key_unknowns=data.get("key_unknowns", [])[:3],
        research_brief=data.get("research_brief", ""),
        suggested_tool_priority=data.get("suggested_tool_priority", []),
        memory_context=data.get("memory_context", ""),
    )
