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
    ResponseCard,
    SearchResult,
    VisionResult,
)

logger = logging.getLogger("lens.orchestrator")

_OVERALL_TIMEOUT_S = 2.5


class LensState(TypedDict):
    input: LensInput
    image_b64: str
    vision_result: VisionResult | None
    memory_result: MemoryResult | None
    search_result: SearchResult | None
    response_card: ResponseCard | None
    cost_log: list[CostEntry]
    errors: list[str]
    _start_time: float
    _cache_key: str


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
        "search_result": None,
        "response_card": None,
        "cost_log": [],
        "errors": [],
        "_start_time": time.monotonic(),
        "_cache_key": cache_key,
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
        card: ResponseCard = FallbackCard(**{k: v for k, v in cached.items()
                                            if k in FallbackCard.model_fields})
    else:
        card = NormalCard(**{k: v for k, v in cached.items()
                             if k in NormalCard.model_fields})
    logger.info("Returning cached card — skipping all agents")
    return {**state, "response_card": card}


def _should_run_agents(state: LensState) -> str:
    return "done" if state["response_card"] is not None else "vision_memory"


async def vision_memory_node(state: LensState) -> LensState:
    vision_result, memory_result = await asyncio.gather(
        _safe_vision(state),
        _safe_memory(state, state["input"].image_path),
    )
    return {**state, "vision_result": vision_result, "memory_result": memory_result}


def _should_search(state: LensState) -> str:
    """Skip Search entirely when Vision has no useful identification."""
    vision = state["vision_result"]
    if vision is None or vision.needs_fallback or vision.confidence_level == "guessing":
        logger.info("Confidence gate: skipping Search (confidence=%s, needs_fallback=%s)",
                    vision.confidence_level if vision else "none",
                    vision.needs_fallback if vision else True)
        return "fuse"
    return "search"


async def search_node(state: LensState) -> LensState:
    search_result = await _safe_search(state, state["vision_result"], state["memory_result"])
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
    asyncio.ensure_future(_write_cache_async(state["_cache_key"], card))
    return {**state, "response_card": card}


async def _write_cache_async(cache_key: str, card: ResponseCard) -> None:
    from src.cache import cache_set
    try:
        await cache_set(cache_key, card.model_dump())
    except Exception as exc:
        logger.warning("cache_set failed: %s", exc)


async def write_memory_node(state: LensState) -> LensState:
    vision = state["vision_result"]
    card = state["response_card"]
    if not vision or not isinstance(card, NormalCard):
        return state

    asyncio.ensure_future(
        _write_memory_async(
            user_id=state["input"].user_id,
            subject_name=vision.entity_name,
            summary=card.body,
            cost_log=state["cost_log"],
            entity_type=vision.entity_type,
        )
    )
    return state


async def _write_memory_async(
    user_id: str, subject_name: str, summary: str, cost_log: list[CostEntry],
    entity_type: str = "unknown",
) -> None:
    import os

    from google import genai

    from src import db
    from src.cost_logger import log_cost

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    try:
        embed_resp = await client.aio.models.embed_content(
            model="text-embedding-004",
            contents=subject_name,
        )
        embedding: list[float] = embed_resp.embeddings[0].values  # type: ignore[index, assignment]
        approx_tokens = max(1, len(subject_name.split()))
        cost_log.append(log_cost("memory_write_embed", "text-embedding-004", approx_tokens, 0))
        await db.write_interaction(
            user_id=user_id,
            subject_name=subject_name,
            summary=summary,
            embedding=embedding,
        )
        await db.upsert_interest(user_id, entity_type)
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
    g.add_node("search", search_node)
    g.add_node("fuse", fuse_node)
    g.add_node("write_memory", write_memory_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "cache_check")
    g.add_conditional_edges("cache_check", _should_run_agents,
                            {"vision_memory": "vision_memory", "done": "write_memory"})
    g.add_conditional_edges("vision_memory", _should_search, {"search": "search", "fuse": "fuse"})
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
        "input": inp, "image_b64": image_b64,
        "vision_result": None, "memory_result": None, "search_result": None,
        "response_card": None, "cost_log": cost_log, "errors": errors,
        "_start_time": start, "_cache_key": "",
    }

    vision_result, memory_result = await asyncio.gather(
        _safe_vision(fake_state),
        _safe_memory(fake_state, inp.image_path),
    )

    search_result = None
    if vision_result and not vision_result.needs_fallback and vision_result.confidence_level != "guessing":
        search_result = await _safe_search(fake_state, vision_result, memory_result)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    cost_usd_so_far = sum(e.cost_usd for e in cost_log)

    final_card = None
    async for chunk, card in stream_fusion(
        vision_result, memory_result, search_result,
        cost_log, cost_usd_so_far, elapsed_ms, inp.user_locale,
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
        "search_result": None,
        "response_card": None,
        "cost_log": [],
        "errors": [],
        "_start_time": time.monotonic(),
        "_cache_key": "",
    }
    return await asyncio.wait_for(
        _graph.ainvoke(initial, config=_run_config(inp)),  # type: ignore[arg-type]
        timeout=_OVERALL_TIMEOUT_S,
    )
