"""LangGraph orchestrator — plan → specialists → fuse → write_memory."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, END

from src.contracts import (
    CostEntry,
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


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def plan_node(state: LensState) -> LensState:
    inp = state["input"]
    image_data = Path(inp.image_path).read_bytes()
    image_b64 = base64.b64encode(image_data).decode()
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


async def specialists_node(state: LensState) -> LensState:
    vision_result, memory_result = await asyncio.gather(
        _safe_vision(state),
        _safe_memory(state, state["input"].image_path),
    )
    search_result = await _safe_search(state, vision_result, memory_result)
    return {**state, "vision_result": vision_result, "memory_result": memory_result, "search_result": search_result}


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
    return {**state, "response_card": card}


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
        )
    )
    return state


async def _write_memory_async(
    user_id: str, subject_name: str, summary: str, cost_log: list[CostEntry]
) -> None:
    import os
    from openai import AsyncOpenAI
    from src.cost_logger import log_cost
    from src import db

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        resp = await client.embeddings.create(model="text-embedding-3-small", input=subject_name)
        cost_log.append(log_cost("memory_write_embed", "text-embedding-3-small", resp.usage.prompt_tokens, 0))
        await db.write_interaction(
            user_id=user_id,
            subject_name=subject_name,
            summary=summary,
            embedding=resp.data[0].embedding,
        )
    except Exception as exc:
        logger.warning("write_memory async failed: %s", exc)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    g = StateGraph(LensState)
    g.add_node("plan", plan_node)
    g.add_node("specialists", specialists_node)
    g.add_node("fuse", fuse_node)
    g.add_node("write_memory", write_memory_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "specialists")
    g.add_edge("specialists", "fuse")
    g.add_edge("fuse", "write_memory")
    g.add_edge("write_memory", END)
    return g


_graph = _build_graph().compile()


async def run_pipeline(inp: LensInput) -> LensState:
    from src import db
    db.init_db()
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
    }
    return await asyncio.wait_for(_graph.ainvoke(initial), timeout=_OVERALL_TIMEOUT_S)
