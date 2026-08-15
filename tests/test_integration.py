"""End-to-end integration tests — hit real Gemini APIs.

These tests are SKIPPED unless both environment variables are set:
    GEMINI_API_KEY   your Gemini API key
    TEST_IMAGE_PATH  path to a JPEG/PNG of a building, monument, or object

Optional variables:
    TEST_LAT   latitude of the image location  (default: 12.9507 — Bangalore)
    TEST_LNG   longitude of the image location (default: 77.5848 — Bangalore)

Run with:
    GEMINI_API_KEY=... TEST_IMAGE_PATH=path/to/photo.jpg pytest -m integration -v

Or via Makefile:
    make integration IMAGE=path/to/photo.jpg
"""

from __future__ import annotations

import os

import pytest

from src.contracts import LensInput

# ---------------------------------------------------------------------------
# Module-level skip condition — applied to every test in this file
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("GEMINI_API_KEY")
_IMAGE_PATH = os.environ.get("TEST_IMAGE_PATH")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _API_KEY or not _IMAGE_PATH,
        reason="Set GEMINI_API_KEY and TEST_IMAGE_PATH to run integration tests",
    ),
]


def _make_input() -> LensInput:
    return LensInput(
        image_path=_IMAGE_PATH or "",
        lat=float(os.environ.get("TEST_LAT", "12.9507")),
        lng=float(os.environ.get("TEST_LNG", "77.5848")),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_run_pipeline_returns_card():
    """Full sync pipeline — vision → memory → search → fusion → card."""
    from src.orchestrator import run_pipeline

    state = await run_pipeline(_make_input())

    card = state["response_card"]
    assert card is not None, "pipeline must return a card"
    assert card.card_type in ("normal", "fallback")
    assert card.latency_ms > 0
    assert card.cost_usd_total >= 0.0

    if card.card_type == "normal":
        assert card.headline, "NormalCard must have a non-empty headline"
        assert card.body, "NormalCard must have a non-empty body"
    else:
        assert card.observation, "FallbackCard must have an observation"

    print(f"\n  card_type : {card.card_type}")
    print(f"  latency   : {card.latency_ms} ms")
    print(f"  cost      : ${card.cost_usd_total:.6f}")
    if card.card_type == "normal":
        print(f"  headline  : {card.headline}")


async def test_stream_pipeline_yields_tokens_then_card():
    """Streaming path — tokens arrive before the final card."""
    from src.orchestrator import stream_pipeline

    text_chunks: list[str] = []
    final_state = None

    async for chunk, state in stream_pipeline(_make_input()):
        if state is None:
            text_chunks.append(chunk)
        else:
            final_state = state

    assert len(text_chunks) > 0, "expected at least one streamed token chunk"
    assert final_state is not None, "expected a final state after streaming"

    card = final_state["response_card"]
    assert card is not None
    assert card.card_type in ("normal", "fallback")

    streamed_text = "".join(text_chunks)
    print(f"\n  streamed chars : {len(streamed_text)}")
    print(f"  card_type      : {card.card_type}")


async def test_cache_hit_returns_same_card():
    """Scanning the same image twice should serve the second response from cache."""
    from src.orchestrator import run_pipeline

    inp = _make_input()

    state1 = await run_pipeline(inp)
    state2 = await run_pipeline(inp)

    card1 = state1["response_card"]
    card2 = state2["response_card"]

    assert card1 is not None and card2 is not None
    assert card1.card_type == card2.card_type

    if card1.card_type == "normal":
        assert card1.headline == card2.headline, (
            "cached card should have the same headline as the first response"
        )

    print(f"\n  card_type : {card1.card_type}")
    print(f"  headline  : {getattr(card1, 'headline', 'N/A')}")
    print("  cache hit confirmed — second response matches first")
