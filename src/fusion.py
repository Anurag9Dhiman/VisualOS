"""Fusion step — Gemini composes agent outputs into a ResponseCard."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from google import genai
from google.genai import types

from src.contracts import (
    Citation,
    CostEntry,
    FallbackCard,
    MemoryResult,
    NormalCard,
    PersonalizedHook,
    ResponseCard,
    SearchResult,
    SourceMix,
    VisionResult,
)
from src.cost_logger import log_cost
from src.prompts import FUSION_SYSTEM_PROMPT, build_fusion_user_message
from src import rate_limiter

logger = logging.getLogger("lens.fusion")

_MODEL = "gemini-2.0-flash"
_TIMEOUT_S = 0.8


def _vision_dict(vision: VisionResult | None) -> dict:
    if vision is None:
        return {"error": "vision agent did not return a result"}
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
            {"subject_name": h.subject_name, "summary": h.summary,
             "similarity_score": h.similarity_score}
            for h in memory.hits
        ],
        "user_interests_snapshot": memory.user_interests_snapshot,
    }


def _search_dict(search: SearchResult | None) -> dict:
    if search is None:
        return {"error": "search agent did not return a result"}
    return {
        "research_plan": search.research_plan,
        "historical_facts": [{"fact": f.fact, "source": f.source} for f in search.historical_facts],
        "live_facts": [{"fact": f.fact, "source": f.source, "as_of": f.as_of} for f in search.live_facts],
        "live_facts_skipped_reason": search.live_facts_skipped_reason,
        "identification_concerns": search.identification_concerns,
        "nearby_context": search.nearby_context,
    }


async def run_fusion(
    vision: VisionResult | None,
    memory: MemoryResult | None,
    search: SearchResult | None,
    cost_log: list[CostEntry],
    cost_usd_total: float,
    latency_ms: int,
    user_locale: str = "en-IN",
) -> ResponseCard:
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    user_msg = build_fusion_user_message(
        _vision_dict(vision), _memory_dict(memory), _search_dict(search), user_locale
    )

    await rate_limiter.acquire(_MODEL)
    try:
        resp = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=_MODEL,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=FUSION_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    max_output_tokens=1200,
                ),
            ),
            timeout=_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("Fusion timed out, returning fallback")
        return FallbackCard(
            headline="Could not compose a response in time.",
            observation="The pipeline exceeded its time budget.",
            suggestion="Try again — this is usually a transient issue.",
            cost_usd_total=cost_usd_total,
            latency_ms=latency_ms,
        )

    usage = resp.usage_metadata
    cost_log.append(log_cost("fusion", _MODEL,
                             usage.prompt_token_count or 0,
                             usage.candidates_token_count or 0))

    try:
        data = json.loads(resp.text)
    except json.JSONDecodeError:
        logger.error("Fusion returned invalid JSON")
        return FallbackCard(
            headline="Could not parse response.",
            observation="Internal error in the fusion step.",
            suggestion="Try again.",
            cost_usd_total=cost_usd_total,
            latency_ms=latency_ms,
        )

    if data.get("card_type") == "fallback":
        return FallbackCard(
            headline=data.get("headline", "Not sure what this is."),
            observation=data.get("observation", ""),
            suggestion=data.get("suggestion", "Try a clearer angle."),
            cost_usd_total=cost_usd_total,
            latency_ms=latency_ms,
        )

    hooks = [
        PersonalizedHook(fact=h["fact"], citation_tag=h.get("citation_tag", ""))
        for h in data.get("personalized_hooks", [])
    ][:3]
    citations = [
        Citation(id=c.get("id", ""), source_name=c.get("source_name", ""),
                 url=c.get("url", ""), as_of=c.get("as_of"))
        for c in data.get("citations", [])
    ]
    sm = data.get("source_mix", {})

    return NormalCard(
        headline=data.get("headline", ""),
        body=data.get("body", ""),
        personalized_hooks=hooks,
        citations=citations,
        confidence_displayed=data.get("confidence_displayed", "high"),
        source_mix=SourceMix(
            used_vision=sm.get("used_vision", True),
            used_memory=sm.get("used_memory", False),
            used_search=sm.get("used_search", True),
        ),
        cost_usd_total=cost_usd_total,
        latency_ms=latency_ms,
    )
