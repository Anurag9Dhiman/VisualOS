"""LangGraph orchestrator — plan → specialists → fuse → write_memory."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from pathlib import Path
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langsmith import traceable

from src.contracts import (
    CostEntry,
    FallbackCard,
    LensInput,
    MemoryResult,
    NormalCard,
    ReasoningTrace,
    ResponseCard,
    SearchResult,
    VisionResult,
)

logger = logging.getLogger("lens.orchestrator")

_OVERALL_TIMEOUT_S = 45.0

# Maps entity_type → which search tools to prioritise first.
_ENTITY_ROUTE: dict[str, str] = {
    "building": "osm_first",
    "monument": "wikidata_first",
    "statue": "wikidata_first",
    "object": "skip",
    "unknown": "default",
}

# Ordered tool lists passed to the Search agent's user message as a priority hint.
_ROUTE_TO_TOOL_PRIORITY: dict[str, list[str]] = {
    "osm_first": ["osm_nearby", "wikipedia_summary", "tavily_search"],
    "wikidata_first": ["wikidata_query", "wikipedia_summary", "tavily_search"],
    "default": ["wikipedia_summary", "wikidata_query", "tavily_search"],
}


class LensState(TypedDict):
    input: LensInput
    image_b64: str
    vision_result: VisionResult | None
    memory_result: MemoryResult | None
    reasoning_trace: ReasoningTrace | None
    search_result: SearchResult | None
    response_card: ResponseCard | None
    cost_log: list[CostEntry]
    errors: list[str]
    _start_time: float
    _cache_key: str
    search_route: str  # "default" | "osm_first" | "wikidata_first" | "skip"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def plan_node(state: LensState) -> LensState:
    inp = state["input"]
    image_data = Path(inp.image_path).read_bytes()
    image_b64 = base64.b64encode(image_data).decode()
    from src.agents.vision import _preprocess_image
    from src.cache import make_cache_key

    preprocessed = _preprocess_image(image_b64)
    cache_key = make_cache_key(preprocessed, inp.lat, inp.lng)
    return {
        **state,
        "image_b64": image_b64,
        "vision_result": None,
        "memory_result": None,
        "reasoning_trace": None,
        "search_result": None,
        "response_card": None,
        "cost_log": [],
        "errors": [],
        "_start_time": time.monotonic(),
        "_cache_key": cache_key,
        "search_route": "default",
    }


async def _safe_vision(state: LensState) -> VisionResult | None:
    from src.agents.vision import run_vision_agent

    try:
        return await run_vision_agent(
            state["image_b64"], state["input"].lat, state["input"].lng, state["cost_log"]
        )
    except Exception as exc:
        logger.warning("Vision agent failed: %s", exc)
        state["errors"].append(f"vision: {exc}")
        return None


async def _safe_memory(state: LensState, subject_name: str) -> MemoryResult | None:
    from src.agents.memory import run_memory_agent

    try:
        return await run_memory_agent(subject_name, state["input"].user_id, state["cost_log"])
    except Exception as exc:
        logger.warning("Memory agent failed: %s", exc)
        state["errors"].append(f"memory: {exc}")
        return None


async def _safe_search(
    state: LensState,
    vision: VisionResult | None,
    memory: MemoryResult | None,
    tool_priority: list[str] | None = None,
    research_brief: str | None = None,
) -> SearchResult | None:
    from src.agents.search import run_search_agent

    entity_name = vision.entity_name if vision else "unknown entity"
    entity_type = vision.entity_type if vision else "unknown"
    confidence_level = vision.confidence_level if vision else "guessing"
    user_interests = list((memory.user_interests_snapshot or {}).keys()) if memory else []
    try:
        return await run_search_agent(
            entity_name=entity_name,
            entity_type=entity_type,
            vision_confidence_level=confidence_level,
            user_interests=user_interests,
            lat=state["input"].lat,
            lng=state["input"].lng,
            cost_log=state["cost_log"],
            tool_priority=tool_priority,
            research_brief=research_brief,
        )
    except Exception as exc:
        logger.warning("Search agent failed: %s", exc)
        state["errors"].append(f"search: {exc}")
        return None


async def cache_check_node(state: LensState) -> LensState:
    from src.cache import cache_get

    cached = await cache_get(state["_cache_key"])
    if cached is None:
        return state
    card_type = cached.get("card_type", "normal")
    if card_type == "fallback":
        card: ResponseCard = FallbackCard(
            **{k: v for k, v in cached.items() if k in FallbackCard.model_fields}
        )
    else:
        card = NormalCard(**{k: v for k, v in cached.items() if k in NormalCard.model_fields})
    logger.info("Returning cached card — skipping all agents")

    # Restore a minimal VisionResult from cached entity metadata so that
    # _store_session in server.py can create a voice session on cache hits.
    vision_from_cache: VisionResult | None = None
    meta = cached.get("_entity_meta")
    if meta:
        try:
            vision_from_cache = VisionResult(
                entity_name=meta["entity_name"],
                entity_type=meta.get("entity_type", "unknown"),
                confidence_level=meta.get("confidence_level", "fairly_sure"),
                evidence=["(restored from cache)"],
                alternatives=[],
                failure_modes_checked=["(restored from cache)"],
                needs_fallback=False,
            )
        except Exception as exc:
            logger.warning("Could not restore VisionResult from cache meta: %s", exc)

    return {**state, "response_card": card, "vision_result": vision_from_cache}


def _should_run_agents(state: LensState) -> str:
    return "done" if state["response_card"] is not None else "vision_memory"


async def vision_memory_node(state: LensState) -> LensState:
    vision_result, memory_result = await asyncio.gather(
        _safe_vision(state),
        _safe_memory(state, state["input"].image_path),
    )
    return {**state, "vision_result": vision_result, "memory_result": memory_result}


async def _safe_reasoning(state: LensState) -> ReasoningTrace | None:
    from src.agents.reasoning import run_reasoning_agent

    try:
        return await run_reasoning_agent(
            state["vision_result"],
            state["memory_result"],
            state["input"].lat,
            state["input"].lng,
            state["cost_log"],
        )
    except Exception as exc:
        logger.warning("Reasoning agent failed: %s", exc)
        state["errors"].append(f"reasoning: {exc}")
        return None


async def reasoning_node(state: LensState) -> LensState:
    reasoning_trace = await _safe_reasoning(state)
    return {**state, "reasoning_trace": reasoning_trace}


def route_node(state: LensState) -> LensState:
    """Set search_route based on Vision confidence + entity_type."""
    vision = state["vision_result"]
    if vision is None or vision.needs_fallback or vision.confidence_level == "guessing":
        logger.info(
            "Route: skip (confidence=%s, needs_fallback=%s)",
            vision.confidence_level if vision else "none",
            vision.needs_fallback if vision else True,
        )
        return {**state, "search_route": "skip"}
    route = _ENTITY_ROUTE.get(vision.entity_type, "default")
    logger.info("Route: %s (entity_type=%s)", route, vision.entity_type)
    return {**state, "search_route": route}


def _should_search_after_route(state: LensState) -> str:
    return "fuse" if state["search_route"] == "skip" else "search"


async def search_node(state: LensState) -> LensState:
    reasoning = state.get("reasoning_trace")
    # Reasoning-suggested priority wins; fall back to static route table.
    tool_priority = (
        reasoning.suggested_tool_priority or _ROUTE_TO_TOOL_PRIORITY.get(state["search_route"])
        if reasoning
        else _ROUTE_TO_TOOL_PRIORITY.get(state["search_route"])
    )
    research_brief = reasoning.research_brief if reasoning else None
    search_result = await _safe_search(
        state, state["vision_result"], state["memory_result"], tool_priority, research_brief
    )
    return {**state, "search_result": search_result}


async def fuse_node(state: LensState) -> LensState:
    from src.fusion import run_fusion

    elapsed_ms = int((time.monotonic() - state["_start_time"]) * 1000)
    cost_usd_total = sum(e.cost_usd for e in state["cost_log"])

    card = await run_fusion(
        state["vision_result"],
        state["memory_result"],
        state["search_result"],
        state["cost_log"],
        cost_usd_total=cost_usd_total,
        latency_ms=elapsed_ms,
        user_locale=state["input"].user_locale,
    )
    asyncio.ensure_future(_write_cache_async(state["_cache_key"], card, state["vision_result"]))
    return {**state, "response_card": card}


async def _write_cache_async(
    cache_key: str, card: ResponseCard, vision: VisionResult | None
) -> None:
    from src.cache import cache_set

    entity_dict: dict | None = None
    if vision is not None:
        entity_dict = {
            "entity_name": vision.entity_name,
            "entity_type": vision.entity_type,
            "confidence_level": vision.confidence_level,
        }
    try:
        await cache_set(cache_key, card.model_dump(), entity_dict)
    except Exception as exc:
        logger.warning("cache_set failed: %s", exc)


async def write_memory_node(state: LensState) -> LensState:
    vision = state["vision_result"]
    card = state["response_card"]
    if not vision or not isinstance(card, NormalCard):
        return state

    search = state.get("search_result")
    asyncio.ensure_future(
        _write_memory_async(
            user_id=state["input"].user_id,
            subject_name=vision.entity_name,
            summary=card.body,
            cost_log=state["cost_log"],
            entity_type=vision.entity_type,
            search_result=search,
        )
    )
    return state


async def _write_memory_async(
    user_id: str,
    subject_name: str,
    summary: str,
    cost_log: list[CostEntry],
    entity_type: str = "unknown",
    search_result=None,
) -> None:
    import os

    from google import genai

    from src import db, memory_l1
    from src.cost_logger import log_cost

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    try:
        embed_resp = await client.aio.models.embed_content(
            model="gemini-embedding-001",
            contents=subject_name,
        )
        embedding: list[float] = embed_resp.embeddings[0].values  # type: ignore[index, assignment]
        approx_tokens = max(1, len(subject_name.split()))
        cost_log.append(log_cost("memory_write_embed", "gemini-embedding-001", approx_tokens, 0))

        # Layer 2: write interaction with embedding
        await db.write_interaction(
            user_id=user_id,
            subject_name=subject_name,
            summary=summary,
            embedding=embedding,
        )
        await db.upsert_interest(user_id, entity_type)

        # Layer 1: record in in-context rolling window
        memory_l1.record_scan(user_id, subject_name, summary)

        # Layer 3: persist facts from Search result
        if search_result is not None:
            facts: list[dict] = []
            for hf in search_result.historical_facts or []:
                facts.append({"fact_key": "historical", "fact_value": hf.fact, "source": hf.source})
            for lf in search_result.live_facts or []:
                facts.append(
                    {
                        "fact_key": "live",
                        "fact_value": lf.fact,
                        "source": lf.source,
                        "as_of": lf.as_of,
                    }
                )
            if facts:
                await db.write_entity_facts(subject_name, facts)

    except Exception as exc:
        logger.warning("write_memory async failed: %s", exc)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def _build_graph() -> StateGraph:
    g = StateGraph(LensState)
    g.add_node("plan", plan_node)
    g.add_node("cache_check", cache_check_node)
    g.add_node("vision_memory", vision_memory_node)
    g.add_node("reasoning", reasoning_node)
    g.add_node("route", route_node)
    g.add_node("search", search_node)
    g.add_node("fuse", fuse_node)
    g.add_node("write_memory", write_memory_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "cache_check")
    g.add_conditional_edges(
        "cache_check",
        _should_run_agents,
        {"vision_memory": "vision_memory", "done": "write_memory"},
    )
    g.add_edge("vision_memory", "reasoning")
    g.add_edge("reasoning", "route")
    g.add_conditional_edges(
        "route", _should_search_after_route, {"search": "search", "fuse": "fuse"}
    )
    g.add_edge("search", "fuse")
    g.add_edge("fuse", "write_memory")
    g.add_edge("write_memory", END)
    return g


_graph = _build_graph().compile()


def _run_config(inp: LensInput) -> RunnableConfig:
    """LangGraph run config — carries per-request metadata into LangSmith traces."""
    return {
        "run_name": "lens_pipeline",
        "metadata": {
            "user_id": inp.user_id,
            "image_path": str(inp.image_path),
            "lat": inp.lat,
            "lng": inp.lng,
        },
        "tags": ["lens-os", "phase0"],
    }


@traceable(name="lens_stream_pipeline", run_type="chain")
async def stream_pipeline(inp: LensInput):
    """Run Vision + Memory + Search, then stream Fusion tokens.

    Yields (chunk, None) for each text chunk, then (full_text, final_state)
    as the last item once the card is fully assembled.
    """
    from src import db
    from src.fusion import stream_fusion

    db.init_db()
    start = time.monotonic()
    cost_log: list[CostEntry] = []
    errors: list[str] = []

    image_data = Path(inp.image_path).read_bytes()
    image_b64 = base64.b64encode(image_data).decode()

    fake_state: LensState = {
        "input": inp,
        "image_b64": image_b64,
        "vision_result": None,
        "memory_result": None,
        "reasoning_trace": None,
        "search_result": None,
        "response_card": None,
        "cost_log": cost_log,
        "errors": errors,
        "_start_time": start,
        "_cache_key": "",
        "search_route": "default",
    }

    vision_result, memory_result = await asyncio.gather(
        _safe_vision(fake_state),
        _safe_memory(fake_state, inp.image_path),
    )
    fake_state = {**fake_state, "vision_result": vision_result, "memory_result": memory_result}

    reasoning_trace = await _safe_reasoning(fake_state)
    fake_state = {**fake_state, "reasoning_trace": reasoning_trace}

    search_result = None
    if (
        vision_result
        and not vision_result.needs_fallback
        and vision_result.confidence_level != "guessing"
    ):
        route = _ENTITY_ROUTE.get(vision_result.entity_type, "default")
        if route != "skip":
            tool_priority = (
                reasoning_trace.suggested_tool_priority or _ROUTE_TO_TOOL_PRIORITY.get(route)
                if reasoning_trace
                else _ROUTE_TO_TOOL_PRIORITY.get(route)
            )
            research_brief = reasoning_trace.research_brief if reasoning_trace else None
            search_result = await _safe_search(
                fake_state, vision_result, memory_result, tool_priority, research_brief
            )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    cost_usd_so_far = sum(e.cost_usd for e in cost_log)

    final_card = None
    async for chunk, card in stream_fusion(
        vision_result,
        memory_result,
        search_result,
        cost_log,
        cost_usd_so_far,
        elapsed_ms,
        inp.user_locale,
    ):
        if card is not None:
            final_card = card
            final_state: LensState = {
                **fake_state,
                "vision_result": vision_result,
                "memory_result": memory_result,
                "search_result": search_result,
                "response_card": final_card,
            }
            yield chunk, final_state
        else:
            yield chunk, None


async def run_pipeline(inp: LensInput) -> LensState:
    from src import db
    from src.cache import init_cache

    db.init_db()
    init_cache(db.DB_PATH)
    initial: LensState = {
        "input": inp,
        "image_b64": "",
        "vision_result": None,
        "memory_result": None,
        "reasoning_trace": None,
        "search_result": None,
        "response_card": None,
        "cost_log": [],
        "errors": [],
        "_start_time": time.monotonic(),
        "_cache_key": "",
        "search_route": "default",
    }
    return await asyncio.wait_for(
        _graph.ainvoke(initial, config=_run_config(inp)),  # type: ignore[arg-type]
        timeout=_OVERALL_TIMEOUT_S,
    )
