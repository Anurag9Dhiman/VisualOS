"""Search Agent — Gemini ReAct loop with 3-call budget."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from google import genai
from google.genai import types

from src import rate_limiter
from src.contracts import (
    CostEntry,
    HistoricalFact,
    LiveFact,
    SearchResult,
    ToolCallRecord,
)
from src.cost_logger import log_cost
from src.prompts import SEARCH_SYSTEM_PROMPT, GeoPoint, build_search_user_message
from src.tools.osm_client import osm_lookup
from src.tools.tavily_client import tavily_search
from src.tools.wikidata_client import wikidata_lookup
from src.tools.wikipedia_client import wikipedia_search

logger = logging.getLogger("lens.search")

_MODEL = "gemini-2.0-flash"
_TIMEOUT_S = 0.8


async def _dispatch_tool(tool_name: str, tool_input: dict) -> str:
    try:
        if tool_name == "wikipedia_summary":
            wiki = await wikipedia_search(tool_input.get("entity", tool_input.get("query", "")))
            return json.dumps({"summary": wiki.extract, "url": wiki.url})
        elif tool_name == "wikidata_query":
            wd = await wikidata_lookup(tool_input.get("entity", "Q1"))
            return json.dumps(wd.facts)
        elif tool_name == "tavily_search":
            tav = await tavily_search(tool_input.get("query", ""))
            return json.dumps(tav.results[:3])
        elif tool_name == "osm_nearby":
            osm = await osm_lookup(
                tool_input.get("lat", 0.0),
                tool_input.get("lng", 0.0),
                tool_input.get("radius_m", 50),
            )
            return json.dumps({"name": osm.name, "address": osm.address,
                               "opening_hours": osm.opening_hours})
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _parse_search_output(data: dict) -> SearchResult:
    tool_calls = [
        ToolCallRecord(
            tool=tc.get("tool", ""),
            input=tc.get("input", {}),
            justification=tc.get("justification", ""),
            observation=tc.get("observation", ""),
        )
        for tc in data.get("tool_calls", [])
    ][:3]

    historical_facts = [
        HistoricalFact(fact=f["fact"], source=f["source"])
        for f in data.get("historical_facts", [])
    ]
    live_facts = [
        LiveFact(fact=f["fact"], source=f["source"], as_of=f.get("as_of", ""))
        for f in data.get("live_facts", [])
    ]
    skipped_reason = data.get("live_facts_skipped_reason", "")
    if not live_facts and not skipped_reason:
        skipped_reason = "No live data retrieved."

    return SearchResult(
        research_plan=data.get("research_plan", ""),
        tool_calls=tool_calls,
        historical_facts=historical_facts,
        live_facts=live_facts,
        live_facts_skipped_reason=skipped_reason,
        identification_concerns=data.get("identification_concerns", []),
        nearby_context=data.get("nearby_context", ""),
    )


async def run_search_agent(
    entity_name: str,
    entity_type: str,
    vision_confidence_level: str,
    user_interests: list[str],
    lat: float | None,
    lng: float | None,
    cost_log: list[CostEntry],
) -> SearchResult:
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    location = GeoPoint(lat=lat or 0.0, lng=lng or 0.0)
    user_msg = build_search_user_message(
        entity_name, entity_type, vision_confidence_level, location, user_interests
    )

    await rate_limiter.acquire(_MODEL)
    try:
        resp = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=_MODEL,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=SEARCH_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    max_output_tokens=1500,
                ),
            ),
            timeout=_TIMEOUT_S,
        )
    except TimeoutError:
        logger.warning("Search agent timed out for '%s'", entity_name)
        return SearchResult(
            research_plan="Timed out.",
            tool_calls=[],
            historical_facts=[],
            live_facts=[],
            live_facts_skipped_reason="Search did not complete within budget.",
        )

    usage = resp.usage_metadata
    cost_log.append(log_cost("search", _MODEL,
                             (usage.prompt_token_count or 0) if usage else 0,
                             (usage.candidates_token_count or 0) if usage else 0))

    try:
        data = json.loads(resp.text or "{}")
    except json.JSONDecodeError:
        logger.error("Search agent returned invalid JSON")
        return SearchResult(
            research_plan="JSON parse error.",
            tool_calls=[],
            historical_facts=[],
            live_facts=[],
            live_facts_skipped_reason="Search output could not be parsed.",
        )

    return _parse_search_output(data)
