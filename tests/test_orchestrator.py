"""Tests for the LangGraph orchestrator nodes and routing.

Covers: _run_config metadata, conditional routing helpers (_should_run_agents,
_should_search), individual node functions, and end-to-end run_pipeline paths
(cache HIT shortcut, confident-vision full run, guessing-vision fallback).
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.contracts import (
    FallbackCard,
    LensInput,
    MemoryResult,
    NormalCard,
    SearchResult,
    VisionResult,
)
from src.orchestrator import (
    _run_config,
    _should_run_agents,
    _should_search,
    cache_check_node,
    plan_node,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg(path: Path) -> None:
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img.save(path, format="JPEG")


def _vision(confidence: str = "certain", needs_fallback: bool = False) -> VisionResult:
    return VisionResult(
        entity_name="Lalbagh Gate",
        entity_type="monument",
        confidence_level=confidence,
        evidence=["stone gate with inscription"],
        alternatives=[],
        failure_modes_checked=["ruled out replica via GPS"],
        needs_fallback=needs_fallback,
    )


def _memory() -> MemoryResult:
    return MemoryResult(hits=[], user_id="anon", user_interests_snapshot={})


def _search() -> SearchResult:
    return SearchResult(
        research_plan="Wikipedia first.",
        tool_calls=[],
        historical_facts=[{"fact": "Built 1890.", "source": "wikipedia"}],
        live_facts=[],
        live_facts_skipped_reason="Ancient monument — no time-sensitive data.",
        identification_concerns=[],
        nearby_context="",
    )


def _normal_card() -> NormalCard:
    return NormalCard(
        headline="Lalbagh Botanical Garden West Gate",
        body="A colonial-era gate in Bangalore.",
        personalized_hooks=[],
        citations=[],
        confidence_displayed="high",
        source_mix={"used_vision": True, "used_memory": False, "used_search": True},
        cost_usd_total=0.001,
        latency_ms=500,
    )


def _fallback_card() -> FallbackCard:
    return FallbackCard(
        headline="Not sure what this is.",
        observation="Weathered stone surface, partially occluded.",
        suggestion="Try a clearer angle or move closer.",
        cost_usd_total=0.0,
        latency_ms=100,
    )


def _base_state(inp: LensInput | None = None) -> dict:
    if inp is None:
        inp = LensInput(image_path="/fake/image.jpg", lat=12.95, lng=77.58)
    return {
        "input": inp,
        "image_b64": "",
        "vision_result": None,
        "memory_result": None,
        "search_result": None,
        "response_card": None,
        "cost_log": [],
        "errors": [],
        "_start_time": 0.0,
        "_cache_key": "abc123def456" * 4,  # 48-char placeholder
    }


# ---------------------------------------------------------------------------
# _should_run_agents
# ---------------------------------------------------------------------------

def test_should_run_agents_no_card():
    state = _base_state()
    assert _should_run_agents(state) == "vision_memory"


def test_should_run_agents_with_card_returns_done():
    state = _base_state()
    state["response_card"] = _normal_card()
    assert _should_run_agents(state) == "done"


# ---------------------------------------------------------------------------
# _should_search (confidence gate)
# ---------------------------------------------------------------------------

def test_should_search_certain_vision():
    state = _base_state()
    state["vision_result"] = _vision("certain")
    assert _should_search(state) == "search"


def test_should_search_fairly_sure_vision():
    state = _base_state()
    state["vision_result"] = _vision("fairly_sure")
    assert _should_search(state) == "search"


def test_should_search_uncertain_vision():
    state = _base_state()
    state["vision_result"] = _vision("uncertain")
    assert _should_search(state) == "search"


def test_should_search_guessing_skips_to_fuse():
    state = _base_state()
    state["vision_result"] = _vision("guessing", needs_fallback=True)
    assert _should_search(state) == "fuse"


def test_should_search_needs_fallback_skips_to_fuse():
    state = _base_state()
    state["vision_result"] = _vision("uncertain", needs_fallback=True)
    assert _should_search(state) == "fuse"


def test_should_search_none_vision_skips_to_fuse():
    state = _base_state()
    state["vision_result"] = None
    assert _should_search(state) == "fuse"


# ---------------------------------------------------------------------------
# plan_node
# ---------------------------------------------------------------------------

def test_plan_node_base64_encodes_jpeg(tmp_path):
    img_path = tmp_path / "scene.jpg"
    _make_jpeg(img_path)
    state = _base_state(LensInput(image_path=str(img_path), lat=12.95, lng=77.58))

    result = plan_node(state)

    decoded = base64.b64decode(result["image_b64"])
    assert decoded[:3] == b"\xff\xd8\xff"  # JPEG magic bytes


def test_plan_node_produces_64char_cache_key(tmp_path):
    img_path = tmp_path / "scene.jpg"
    _make_jpeg(img_path)
    state = _base_state(LensInput(image_path=str(img_path), lat=12.95, lng=77.58))

    result = plan_node(state)

    assert len(result["_cache_key"]) == 64
    assert all(c in "0123456789abcdef" for c in result["_cache_key"])


def test_plan_node_resets_cost_log(tmp_path):
    img_path = tmp_path / "scene.jpg"
    _make_jpeg(img_path)
    state = _base_state(LensInput(image_path=str(img_path), lat=12.95, lng=77.58))
    state["cost_log"] = [MagicMock()]

    result = plan_node(state)

    assert result["cost_log"] == []


def test_plan_node_same_image_same_key(tmp_path):
    img_path = tmp_path / "scene.jpg"
    _make_jpeg(img_path)
    inp = LensInput(image_path=str(img_path), lat=12.95, lng=77.58)
    state = _base_state(inp)

    r1 = plan_node(state)
    r2 = plan_node(state)
    assert r1["_cache_key"] == r2["_cache_key"]


# ---------------------------------------------------------------------------
# cache_check_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_check_miss_leaves_card_none():
    state = _base_state()
    with patch("src.cache.cache_get", new=AsyncMock(return_value=None)):
        result = await cache_check_node(state)
    assert result["response_card"] is None


@pytest.mark.asyncio
async def test_cache_check_hit_sets_normal_card():
    state = _base_state()
    cached = _normal_card().model_dump()
    with patch("src.cache.cache_get", new=AsyncMock(return_value=cached)):
        result = await cache_check_node(state)
    assert isinstance(result["response_card"], NormalCard)
    assert result["response_card"].headline == "Lalbagh Botanical Garden West Gate"


@pytest.mark.asyncio
async def test_cache_check_hit_sets_fallback_card():
    state = _base_state()
    cached = _fallback_card().model_dump()
    with patch("src.cache.cache_get", new=AsyncMock(return_value=cached)):
        result = await cache_check_node(state)
    assert isinstance(result["response_card"], FallbackCard)
    assert "Not sure" in result["response_card"].headline


# ---------------------------------------------------------------------------
# run_pipeline — end-to-end paths (all agents mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_pipeline_cache_hit_skips_agents(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    img_path = tmp_path / "scene.jpg"
    _make_jpeg(img_path)
    inp = LensInput(image_path=str(img_path), lat=12.95, lng=77.58)

    with patch("src.cache.cache_get", new=AsyncMock(return_value=_normal_card().model_dump())), \
         patch("src.agents.vision.run_vision_agent", new=AsyncMock()) as mock_vis, \
         patch("src.agents.memory.run_memory_agent", new=AsyncMock()) as mock_mem:
        from src.orchestrator import run_pipeline
        state = await run_pipeline(inp)

    mock_vis.assert_not_called()
    mock_mem.assert_not_called()
    assert isinstance(state["response_card"], NormalCard)


@pytest.mark.asyncio
async def test_run_pipeline_confident_vision_calls_search(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    img_path = tmp_path / "scene.jpg"
    _make_jpeg(img_path)
    inp = LensInput(image_path=str(img_path), lat=12.95, lng=77.58)

    with patch("src.cache.cache_get", new=AsyncMock(return_value=None)), \
         patch("src.agents.vision.run_vision_agent", new=AsyncMock(return_value=_vision("certain"))), \
         patch("src.agents.memory.run_memory_agent", new=AsyncMock(return_value=_memory())), \
         patch("src.agents.search.run_search_agent", new=AsyncMock(return_value=_search())) as mock_search, \
         patch("src.fusion.run_fusion", new=AsyncMock(return_value=_normal_card())), \
         patch("src.orchestrator._write_cache_async", new=AsyncMock()), \
         patch("src.orchestrator._write_memory_async", new=AsyncMock()):
        from src.orchestrator import run_pipeline
        state = await run_pipeline(inp)

    mock_search.assert_called_once()
    assert isinstance(state["response_card"], NormalCard)
    assert state["vision_result"].entity_name == "Lalbagh Gate"


@pytest.mark.asyncio
async def test_run_pipeline_guessing_vision_skips_search(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    img_path = tmp_path / "scene.jpg"
    _make_jpeg(img_path)
    inp = LensInput(image_path=str(img_path), lat=12.95, lng=77.58)

    with patch("src.cache.cache_get", new=AsyncMock(return_value=None)), \
         patch("src.agents.vision.run_vision_agent", new=AsyncMock(return_value=_vision("guessing", True))), \
         patch("src.agents.memory.run_memory_agent", new=AsyncMock(return_value=_memory())), \
         patch("src.agents.search.run_search_agent", new=AsyncMock(return_value=_search())) as mock_search, \
         patch("src.fusion.run_fusion", new=AsyncMock(return_value=_fallback_card())), \
         patch("src.orchestrator._write_cache_async", new=AsyncMock()), \
         patch("src.orchestrator._write_memory_async", new=AsyncMock()):
        from src.orchestrator import run_pipeline
        state = await run_pipeline(inp)

    mock_search.assert_not_called()
    assert isinstance(state["response_card"], FallbackCard)


# ---------------------------------------------------------------------------
# _run_config
# ---------------------------------------------------------------------------

def test_run_config_run_name():
    inp = LensInput(image_path="/photos/test.jpg", lat=0.0, lng=0.0)
    assert _run_config(inp)["run_name"] == "lens_pipeline"


def test_run_config_embeds_user_id():
    inp = LensInput(image_path="/photos/test.jpg", lat=0.0, lng=0.0, user_id="user-42")
    assert _run_config(inp)["metadata"]["user_id"] == "user-42"


def test_run_config_embeds_lat_lng():
    inp = LensInput(image_path="/photos/test.jpg", lat=12.95, lng=77.58)
    cfg = _run_config(inp)
    assert cfg["metadata"]["lat"] == 12.95
    assert cfg["metadata"]["lng"] == 77.58


def test_run_config_has_phase0_tag():
    inp = LensInput(image_path="/photos/test.jpg", lat=0.0, lng=0.0)
    assert "phase0" in _run_config(inp)["tags"]


# ---------------------------------------------------------------------------
# stream_pipeline — primary streaming path
# ---------------------------------------------------------------------------

async def _fake_stream_fusion(vision, memory, search, cost_log, cost_usd, latency, locale="en-IN"):
    yield "The Eiffel", None
    yield " Tower.", None
    card = NormalCard(
        headline="The Eiffel Tower",
        body="An iron lattice tower.",
        personalized_hooks=[],
        citations=[],
        confidence_displayed="high",
        source_mix={"used_vision": True, "used_memory": False, "used_search": True},
        cost_usd_total=cost_usd,
        latency_ms=latency,
    )
    yield "The Eiffel Tower.", card


@pytest.mark.asyncio
async def test_stream_pipeline_yields_chunks_then_final_state(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    img_path = tmp_path / "scene.jpg"
    _make_jpeg(img_path)
    inp = LensInput(image_path=str(img_path), lat=12.95, lng=77.58)

    with patch("src.agents.vision.run_vision_agent", new=AsyncMock(return_value=_vision("certain"))), \
         patch("src.agents.memory.run_memory_agent", new=AsyncMock(return_value=_memory())), \
         patch("src.agents.search.run_search_agent", new=AsyncMock(return_value=_search())), \
         patch("src.fusion.stream_fusion", side_effect=_fake_stream_fusion):
        from src.orchestrator import stream_pipeline
        results = []
        async for chunk, state in stream_pipeline(inp):
            results.append((chunk, state))

    # Intermediate chunks have state=None
    intermediate = [(c, s) for c, s in results if s is None]
    assert len(intermediate) >= 1

    # Final item has a populated LensState
    final_chunk, final_state = results[-1]
    assert final_state is not None
    assert isinstance(final_state["response_card"], NormalCard)
    assert final_state["vision_result"].entity_name == "Lalbagh Gate"


@pytest.mark.asyncio
async def test_stream_pipeline_guessing_vision_skips_search(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    img_path = tmp_path / "scene.jpg"
    _make_jpeg(img_path)
    inp = LensInput(image_path=str(img_path), lat=12.95, lng=77.58)

    async def _fallback_stream(vision, memory, search, cost_log, cost_usd, latency, locale="en-IN"):
        yield "", _fallback_card()

    with patch("src.agents.vision.run_vision_agent", new=AsyncMock(return_value=_vision("guessing", True))), \
         patch("src.agents.memory.run_memory_agent", new=AsyncMock(return_value=_memory())), \
         patch("src.agents.search.run_search_agent", new=AsyncMock()) as mock_search, \
         patch("src.fusion.stream_fusion", side_effect=_fallback_stream):
        from src.orchestrator import stream_pipeline
        results = []
        async for chunk, state in stream_pipeline(inp):
            results.append((chunk, state))

    mock_search.assert_not_called()
    _, final_state = results[-1]
    assert isinstance(final_state["response_card"], FallbackCard)


@pytest.mark.asyncio
async def test_stream_pipeline_passes_search_result_when_confident(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    img_path = tmp_path / "scene.jpg"
    _make_jpeg(img_path)
    inp = LensInput(image_path=str(img_path), lat=12.95, lng=77.58)

    captured_search = {}

    async def _capture_stream(vision, memory, search, cost_log, cost_usd, latency, locale="en-IN"):
        captured_search["result"] = search
        yield "", _normal_card()

    with patch("src.agents.vision.run_vision_agent", new=AsyncMock(return_value=_vision("certain"))), \
         patch("src.agents.memory.run_memory_agent", new=AsyncMock(return_value=_memory())), \
         patch("src.agents.search.run_search_agent", new=AsyncMock(return_value=_search())), \
         patch("src.fusion.stream_fusion", side_effect=_capture_stream):
        from src.orchestrator import stream_pipeline
        async for _ in stream_pipeline(inp):
            pass

    assert captured_search["result"] is not None
    assert captured_search["result"].research_plan == "Wikipedia first."
