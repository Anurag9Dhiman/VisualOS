"""Agent integration tests — all Gemini calls are mocked.

Tests verify that each agent correctly:
  - Parses valid API responses into the right contract type
  - Handles timeouts with a safe fallback
  - Handles malformed/empty JSON without crashing
  - Populates cost_log with an entry after a successful call
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.contracts import (
    FallbackCard,
    MemoryHit,
    NormalCard,
    SearchResult,
    VisionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_b64(width: int = 64, height: int = 64) -> str:
    """Return a base64-encoded JPEG for tests that need real image bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(100, 149, 237)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _mock_genai_response(text: str, in_tokens: int = 100, out_tokens: int = 50):
    """Build a mock GenerateContent response."""
    resp = MagicMock()
    resp.text = text
    resp.usage_metadata = MagicMock(
        prompt_token_count=in_tokens,
        candidates_token_count=out_tokens,
    )
    return resp


def _mock_embed_response(dim: int = 8):
    """Build a mock embed_content response."""
    resp = MagicMock()
    embedding = MagicMock()
    embedding.values = [0.5] * dim
    resp.embeddings = [embedding]
    return resp


# ---------------------------------------------------------------------------
# Vision Agent
# ---------------------------------------------------------------------------

_VALID_VISION_JSON = {
    "entity_name": "Eiffel Tower",
    "entity_type": "monument",
    "confidence_level": "certain",
    "evidence": ["iron lattice structure", "Paris skyline context"],
    "alternatives": [],
    "failure_modes_checked": ["lighting adequate", "no occlusion"],
    "needs_fallback": False,
}


@pytest.mark.asyncio
async def test_vision_agent_success(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.agents.vision import run_vision_agent

    mock_resp = _mock_genai_response(json.dumps(_VALID_VISION_JSON))

    with patch("src.agents.vision.genai.Client") as mock_client, \
         patch("src.agents.vision.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        cost_log = []
        result = await run_vision_agent(_make_image_b64(), 48.8584, 2.2945, cost_log)

    assert isinstance(result, VisionResult)
    assert result.entity_name == "Eiffel Tower"
    assert result.confidence_level == "certain"
    assert result.needs_fallback is False
    assert result.confidence_score == 0.95
    assert len(cost_log) == 1
    assert cost_log[0].agent == "vision"


@pytest.mark.asyncio
async def test_vision_agent_guessing_sets_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.agents.vision import run_vision_agent

    guessing = {**_VALID_VISION_JSON,
                "confidence_level": "guessing", "needs_fallback": True}
    mock_resp = _mock_genai_response(json.dumps(guessing))

    with patch("src.agents.vision.genai.Client") as mock_client, \
         patch("src.agents.vision.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        result = await run_vision_agent(_make_image_b64(), None, None, [])

    assert result.needs_fallback is True
    assert result.confidence_score == 0.20


@pytest.mark.asyncio
async def test_vision_agent_timeout_propagates(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.agents.vision import run_vision_agent

    async def _slow(*_, **__):
        await asyncio.sleep(10)

    with patch("src.agents.vision.genai.Client") as mock_client, \
         patch("src.agents.vision.rate_limiter.acquire", new=AsyncMock()), \
         patch("src.agents.vision._TIMEOUT_S", 0.01):
        mock_client.return_value.aio.models.generate_content = _slow

        with pytest.raises(asyncio.TimeoutError):
            await run_vision_agent(_make_image_b64(), None, None, [])


@pytest.mark.asyncio
async def test_vision_agent_logs_cost(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.agents.vision import run_vision_agent

    mock_resp = _mock_genai_response(json.dumps(_VALID_VISION_JSON), in_tokens=200, out_tokens=80)

    with patch("src.agents.vision.genai.Client") as mock_client, \
         patch("src.agents.vision.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        cost_log = []
        await run_vision_agent(_make_image_b64(), None, None, cost_log)

    assert cost_log[0].input_tokens == 200
    assert cost_log[0].output_tokens == 80
    assert cost_log[0].cost_usd > 0


# ---------------------------------------------------------------------------
# Memory Agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_agent_returns_hits(monkeypatch, tmp_db):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.agents.memory import run_memory_agent
    from src import db

    # Pre-seed the db with a high-similarity interaction
    embed = [0.5] * 8
    await db.write_interaction("u1", "Eiffel Tower", "Famous iron tower.", embed)

    with patch("src.agents.memory.genai.Client") as mock_client, \
         patch("src.agents.memory.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.embed_content = AsyncMock(
            return_value=_mock_embed_response(dim=8)
        )

        cost_log = []
        result = await run_memory_agent("Eiffel Tower", "u1", cost_log)

    assert result.user_id == "u1"
    assert len(result.hits) == 1
    assert result.hits[0].subject_name == "Eiffel Tower"
    assert result.hits[0].similarity_score == pytest.approx(1.0, rel=1e-3)
    assert len(cost_log) == 1
    assert cost_log[0].agent == "memory_embed"


@pytest.mark.asyncio
async def test_memory_agent_filters_low_similarity(monkeypatch, tmp_db):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.agents.memory import run_memory_agent
    from src import db

    # Seed with an orthogonal embedding — similarity will be ~0
    await db.write_interaction("u1", "Red Fort", "Mughal fort.", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Query with a perpendicular vector
    perp_embed = MagicMock()
    perp_embed.values = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mock_embed_resp = MagicMock()
    mock_embed_resp.embeddings = [perp_embed]

    with patch("src.agents.memory.genai.Client") as mock_client, \
         patch("src.agents.memory.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.embed_content = AsyncMock(return_value=mock_embed_resp)

        result = await run_memory_agent("Colosseum", "u1", [])

    # Score ~0.0, below 0.75 threshold — should be filtered out
    assert result.hits == []


@pytest.mark.asyncio
async def test_memory_agent_empty_db(monkeypatch, tmp_db):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.agents.memory import run_memory_agent

    with patch("src.agents.memory.genai.Client") as mock_client, \
         patch("src.agents.memory.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.embed_content = AsyncMock(
            return_value=_mock_embed_response()
        )
        result = await run_memory_agent("Anything", "new-user", [])

    assert result.hits == []
    assert result.user_id == "new-user"


# ---------------------------------------------------------------------------
# Search Agent
# ---------------------------------------------------------------------------

_VALID_SEARCH_JSON = {
    "research_plan": "Use Wikipedia for history, Tavily for live info.",
    "tool_calls": [
        {"tool": "wikipedia_summary", "input": {"entity": "Eiffel Tower"},
         "justification": "Get canonical history.", "observation": "Got founding date."}
    ],
    "historical_facts": [{"fact": "Built 1887–1889.", "source": "Wikipedia"}],
    "live_facts": [],
    "live_facts_skipped_reason": "No time-sensitive info needed.",
    "identification_concerns": [],
    "nearby_context": "",
}


@pytest.mark.asyncio
async def test_search_agent_success(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.agents.search import run_search_agent

    mock_resp = _mock_genai_response(json.dumps(_VALID_SEARCH_JSON))

    with patch("src.agents.search.genai.Client") as mock_client, \
         patch("src.agents.search.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        cost_log = []
        result = await run_search_agent(
            entity_name="Eiffel Tower",
            entity_type="monument",
            vision_confidence_level="certain",
            user_interests=["architecture"],
            lat=48.8584, lng=2.2945,
            cost_log=cost_log,
        )

    assert isinstance(result, SearchResult)
    assert len(result.historical_facts) == 1
    assert result.historical_facts[0].fact == "Built 1887–1889."
    assert result.tool_call_count == 1
    assert len(cost_log) == 1
    assert cost_log[0].agent == "search"


@pytest.mark.asyncio
async def test_search_agent_timeout_returns_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.agents.search import run_search_agent

    async def _slow(*_, **__):
        await asyncio.sleep(10)

    with patch("src.agents.search.genai.Client") as mock_client, \
         patch("src.agents.search.rate_limiter.acquire", new=AsyncMock()), \
         patch("src.agents.search._TIMEOUT_S", 0.01):
        mock_client.return_value.aio.models.generate_content = _slow

        result = await run_search_agent(
            entity_name="X", entity_type="building",
            vision_confidence_level="certain",
            user_interests=[], lat=None, lng=None, cost_log=[],
        )

    assert isinstance(result, SearchResult)
    assert "Timed out" in result.research_plan
    assert result.historical_facts == []


@pytest.mark.asyncio
async def test_search_agent_invalid_json_returns_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.agents.search import run_search_agent

    mock_resp = _mock_genai_response("not valid json at all")

    with patch("src.agents.search.genai.Client") as mock_client, \
         patch("src.agents.search.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        result = await run_search_agent(
            entity_name="X", entity_type="building",
            vision_confidence_level="certain",
            user_interests=[], lat=None, lng=None, cost_log=[],
        )

    assert "parse error" in result.research_plan.lower()
    assert result.historical_facts == []


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

_VALID_NORMAL_CARD_JSON = {
    "card_type": "normal",
    "headline": "The Eiffel Tower is an iron lattice tower in Paris.",
    "body": "Built between 1887 and 1889, it stands 330 metres tall.",
    "personalized_hooks": [
        {"fact": "Designed by Gustave Eiffel.", "citation_tag": "wiki"}
    ],
    "citations": [
        {"id": "wiki", "source_name": "Wikipedia",
         "url": "https://en.wikipedia.org/wiki/Eiffel_Tower", "as_of": None}
    ],
    "confidence_displayed": "high",
    "source_mix": {"used_vision": True, "used_memory": False, "used_search": True},
}

_VALID_FALLBACK_JSON = {
    "card_type": "fallback",
    "headline": "Not sure what this is.",
    "observation": "Partially occluded stone structure.",
    "suggestion": "Try a clearer angle.",
}


@pytest.mark.asyncio
async def test_fusion_returns_normal_card(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.fusion import run_fusion
    from src.contracts import VisionResult, MemoryResult

    vision = VisionResult(**_VALID_VISION_JSON)
    memory = MemoryResult(hits=[], user_id="u1")
    mock_resp = _mock_genai_response(json.dumps(_VALID_NORMAL_CARD_JSON))

    with patch("src.fusion.genai.Client") as mock_client, \
         patch("src.fusion.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        card = await run_fusion(vision, memory, None, [], cost_usd_total=0.0, latency_ms=500)

    assert isinstance(card, NormalCard)
    assert card.headline == _VALID_NORMAL_CARD_JSON["headline"]
    assert card.confidence_displayed == "high"
    assert len(card.personalized_hooks) == 1
    assert card.source_mix.used_vision is True
    assert card.latency_ms == 500


@pytest.mark.asyncio
async def test_fusion_returns_fallback_card(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.fusion import run_fusion

    mock_resp = _mock_genai_response(json.dumps(_VALID_FALLBACK_JSON))

    with patch("src.fusion.genai.Client") as mock_client, \
         patch("src.fusion.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        card = await run_fusion(None, None, None, [], cost_usd_total=0.001, latency_ms=800)

    assert isinstance(card, FallbackCard)
    assert card.headline == "Not sure what this is."
    assert card.suggestion == "Try a clearer angle."
    assert card.cost_usd_total == pytest.approx(0.001)


@pytest.mark.asyncio
async def test_fusion_timeout_returns_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.fusion import run_fusion

    async def _slow(*_, **__):
        await asyncio.sleep(10)

    with patch("src.fusion.genai.Client") as mock_client, \
         patch("src.fusion.rate_limiter.acquire", new=AsyncMock()), \
         patch("src.fusion._TIMEOUT_S", 0.01):
        mock_client.return_value.aio.models.generate_content = _slow

        card = await run_fusion(None, None, None, [], cost_usd_total=0.0, latency_ms=0)

    assert isinstance(card, FallbackCard)
    assert "time" in card.headline.lower() or "time" in card.observation.lower()


@pytest.mark.asyncio
async def test_fusion_invalid_json_returns_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.fusion import run_fusion

    mock_resp = _mock_genai_response("{bad json}")

    with patch("src.fusion.genai.Client") as mock_client, \
         patch("src.fusion.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        card = await run_fusion(None, None, None, [], cost_usd_total=0.0, latency_ms=0)

    assert isinstance(card, FallbackCard)


# ---------------------------------------------------------------------------
# _dispatch_tool — search agent tool dispatcher
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_tool_wikipedia():
    from src.agents.search import _dispatch_tool
    from src.contracts import WikipediaResult

    mock_result = WikipediaResult(title="Eiffel Tower", extract="Iron lattice tower.", url="https://en.wikipedia.org/wiki/Eiffel_Tower")

    with patch("src.agents.search.wikipedia_search", new=AsyncMock(return_value=mock_result)):
        out = await _dispatch_tool("wikipedia_summary", {"entity": "Eiffel Tower"})

    data = json.loads(out)
    assert "Iron lattice tower" in data["summary"]
    assert "wikipedia.org" in data["url"]


@pytest.mark.asyncio
async def test_dispatch_tool_wikidata():
    from src.agents.search import _dispatch_tool
    from src.contracts import WikidataResult

    mock_result = WikidataResult(entity_id="Q243", label="Eiffel Tower", facts={"height": "330 m"}, url="https://www.wikidata.org/wiki/Q243")

    with patch("src.agents.search.wikidata_lookup", new=AsyncMock(return_value=mock_result)):
        out = await _dispatch_tool("wikidata_query", {"entity": "Q243"})

    data = json.loads(out)
    assert data["height"] == "330 m"


@pytest.mark.asyncio
async def test_dispatch_tool_tavily():
    from src.agents.search import _dispatch_tool
    from src.contracts import TavilyResult

    mock_result = TavilyResult(query="Eiffel Tower hours", results=[{"title": "Hours", "url": "https://example.com", "content": "Open daily."}])

    with patch("src.agents.search.tavily_search", new=AsyncMock(return_value=mock_result)):
        out = await _dispatch_tool("tavily_search", {"query": "Eiffel Tower hours"})

    data = json.loads(out)
    assert data[0]["title"] == "Hours"


@pytest.mark.asyncio
async def test_dispatch_tool_osm():
    from src.agents.search import _dispatch_tool
    from src.contracts import OSMResult

    mock_result = OSMResult(name="Lalbagh Gate", address="Lalbagh Road, Bangalore", opening_hours="06:00-19:00", wheelchair=None)

    with patch("src.agents.search.osm_lookup", new=AsyncMock(return_value=mock_result)):
        out = await _dispatch_tool("osm_nearby", {"lat": 12.95, "lng": 77.58, "radius_m": 50})

    data = json.loads(out)
    assert data["name"] == "Lalbagh Gate"
    assert "Bangalore" in data["address"]


@pytest.mark.asyncio
async def test_dispatch_tool_unknown_returns_error():
    from src.agents.search import _dispatch_tool

    out = await _dispatch_tool("nonexistent_tool", {})
    data = json.loads(out)
    assert "error" in data
    assert "Unknown tool" in data["error"]


@pytest.mark.asyncio
async def test_dispatch_tool_exception_returns_error_json():
    from src.agents.search import _dispatch_tool

    with patch("src.agents.search.wikipedia_search", new=AsyncMock(side_effect=RuntimeError("network down"))):
        out = await _dispatch_tool("wikipedia_summary", {"entity": "anything"})

    data = json.loads(out)
    assert "error" in data
    assert "network down" in data["error"]
