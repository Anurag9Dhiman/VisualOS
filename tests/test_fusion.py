"""Tests for fusion helpers — _parse_card and stream_fusion.

run_fusion is already covered by test_agents.py. These tests target
the streaming path and the shared _parse_card helper that both paths use.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.contracts import FallbackCard, NormalCard


# ---------------------------------------------------------------------------
# _parse_card — shared JSON → ResponseCard parser
# ---------------------------------------------------------------------------

_NORMAL_JSON = json.dumps({
    "card_type": "normal",
    "headline": "The Eiffel Tower",
    "body": "An iron lattice tower in Paris.",
    "personalized_hooks": [
        {"fact": "Designed by Gustave Eiffel.", "citation_tag": "wiki"}
    ],
    "citations": [{"id": "wiki", "source_name": "Wikipedia",
                   "url": "https://en.wikipedia.org/wiki/Eiffel_Tower", "as_of": None}],
    "confidence_displayed": "high",
    "source_mix": {"used_vision": True, "used_memory": False, "used_search": True},
})

_FALLBACK_JSON = json.dumps({
    "card_type": "fallback",
    "headline": "Not sure what this is.",
    "observation": "Low-contrast image.",
    "suggestion": "Move closer.",
})


def test_parse_card_normal():
    from src.fusion import _parse_card
    card = _parse_card(_NORMAL_JSON, cost_usd_total=0.002, latency_ms=600)
    assert isinstance(card, NormalCard)
    assert card.headline == "The Eiffel Tower"
    assert card.confidence_displayed == "high"
    assert card.cost_usd_total == pytest.approx(0.002)
    assert card.latency_ms == 600
    assert len(card.personalized_hooks) == 1
    assert card.personalized_hooks[0].fact == "Designed by Gustave Eiffel."


def test_parse_card_fallback():
    from src.fusion import _parse_card
    card = _parse_card(_FALLBACK_JSON, cost_usd_total=0.0, latency_ms=200)
    assert isinstance(card, FallbackCard)
    assert card.headline == "Not sure what this is."
    assert card.observation == "Low-contrast image."
    assert card.suggestion == "Move closer."


def test_parse_card_invalid_json_returns_fallback():
    from src.fusion import _parse_card
    card = _parse_card("not json {{", cost_usd_total=0.0, latency_ms=0)
    assert isinstance(card, FallbackCard)
    assert "parse" in card.headline.lower() or "error" in card.observation.lower()


def test_parse_card_caps_hooks_at_3():
    from src.fusion import _parse_card
    data = json.loads(_NORMAL_JSON)
    data["personalized_hooks"] = [
        {"fact": f"fact {i}", "citation_tag": ""} for i in range(5)
    ]
    card = _parse_card(json.dumps(data), cost_usd_total=0.0, latency_ms=0)
    assert isinstance(card, NormalCard)
    assert len(card.personalized_hooks) == 3


def test_parse_card_missing_source_mix_defaults_to_vision_true():
    from src.fusion import _parse_card
    data = json.loads(_NORMAL_JSON)
    del data["source_mix"]
    card = _parse_card(json.dumps(data), cost_usd_total=0.0, latency_ms=0)
    assert isinstance(card, NormalCard)
    assert card.source_mix.used_vision is True


# ---------------------------------------------------------------------------
# stream_fusion — async generator
# ---------------------------------------------------------------------------

def _make_stream_chunk(text: str | None, prompt_tokens: int = 0, cand_tokens: int = 0):
    chunk = MagicMock()
    chunk.text = text
    if cand_tokens:
        chunk.usage_metadata = MagicMock(
            prompt_token_count=prompt_tokens,
            candidates_token_count=cand_tokens,
        )
    else:
        chunk.usage_metadata = None
    return chunk


async def _fake_stream(chunks):
    """Async generator that yields chunks, simulating generate_content_stream."""
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_stream_fusion_yields_chunks_then_card(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.fusion import stream_fusion

    parts = ['"card_type": "normal"', ', "headline": "Eiffel"', ', "body": "tall tower."']
    full_json = json.dumps({
        "card_type": "normal",
        "headline": "Eiffel",
        "body": "tall tower.",
        "personalized_hooks": [],
        "citations": [],
        "confidence_displayed": "high",
        "source_mix": {"used_vision": True, "used_memory": False, "used_search": True},
    })
    chunks = [
        _make_stream_chunk(p) for p in ["{", *parts, "}"]
    ]
    # Last chunk carries usage metadata
    chunks[-1].usage_metadata = MagicMock(prompt_token_count=50, candidates_token_count=30)

    mock_stream = _fake_stream(chunks)

    with patch("src.fusion.genai.Client") as mock_client, \
         patch("src.fusion.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.generate_content_stream = AsyncMock(
            return_value=mock_stream
        )

        results = []
        async for text_chunk, card in stream_fusion(
            None, None, None, [], cost_usd_total=0.0, latency_ms=100
        ):
            results.append((text_chunk, card))

    # All intermediate yields have card=None
    intermediate = [(t, c) for t, c in results if c is None]
    assert len(intermediate) > 0

    # Final yield has card set
    final_text, final_card = results[-1]
    assert final_card is not None


@pytest.mark.asyncio
async def test_stream_fusion_logs_cost_from_last_chunk(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.fusion import stream_fusion

    full_json = json.dumps({
        "card_type": "normal", "headline": "h", "body": "b",
        "personalized_hooks": [], "citations": [],
        "confidence_displayed": "high",
        "source_mix": {"used_vision": True, "used_memory": False, "used_search": False},
    })
    last_chunk = _make_stream_chunk(full_json, prompt_tokens=80, cand_tokens=40)
    last_chunk.usage_metadata = MagicMock(prompt_token_count=80, candidates_token_count=40)

    with patch("src.fusion.genai.Client") as mock_client, \
         patch("src.fusion.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.generate_content_stream = AsyncMock(
            return_value=_fake_stream([last_chunk])
        )

        cost_log = []
        async for _ in stream_fusion(None, None, None, cost_log, 0.0, 0):
            pass

    assert len(cost_log) == 1
    assert cost_log[0].agent == "fusion_stream"
    assert cost_log[0].input_tokens == 80
    assert cost_log[0].output_tokens == 40


@pytest.mark.asyncio
async def test_stream_fusion_error_yields_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from src.fusion import stream_fusion

    async def _error_stream():
        raise RuntimeError("upstream blew up")
        yield  # make it a generator

    with patch("src.fusion.genai.Client") as mock_client, \
         patch("src.fusion.rate_limiter.acquire", new=AsyncMock()):
        mock_client.return_value.aio.models.generate_content_stream = AsyncMock(
            return_value=_error_stream()
        )

        results = []
        async for text_chunk, card in stream_fusion(
            None, None, None, [], 0.0, 0
        ):
            results.append((text_chunk, card))

    # Should yield exactly one item: the fallback card
    assert len(results) == 1
    _, fallback = results[0]
    assert isinstance(fallback, FallbackCard)
    assert "Streaming failed" in fallback.headline
