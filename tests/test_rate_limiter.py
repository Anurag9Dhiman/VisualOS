"""Tests for the sliding-window rate limiter — no API calls needed."""

from __future__ import annotations

import asyncio
import time

import pytest

from src import rate_limiter


# ---------------------------------------------------------------------------
# Basic slot acquisition
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acquire_within_limit_is_instant():
    rate_limiter.set_limit("test-model", 5)
    start = time.monotonic()
    for _ in range(5):
        await rate_limiter.acquire("test-model")
    elapsed = time.monotonic() - start
    # All 5 should complete well under 0.1s (no sleeping needed)
    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_acquire_records_timestamps():
    rate_limiter.set_limit("test-model", 10)
    for _ in range(3):
        await rate_limiter.acquire("test-model")
    assert len(rate_limiter._windows.get("test-model", [])) == 3


# ---------------------------------------------------------------------------
# Rate limit enforcement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exceeding_limit_causes_wait(monkeypatch):
    """Fill the window, then verify the next call sleeps."""
    rate_limiter.set_limit("slow-model", 2)
    slept: list[float] = []

    original_sleep = asyncio.sleep

    async def mock_sleep(s: float) -> None:
        slept.append(s)
        # Don't actually sleep — just advance time by manipulating window timestamps
        window = rate_limiter._windows.get("slow-model")
        if window:
            # Evict the oldest entry to simulate time passing
            window.popleft()

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    # Fill the 2-slot window
    await rate_limiter.acquire("slow-model")
    await rate_limiter.acquire("slow-model")

    # Third call should trigger sleep
    await rate_limiter.acquire("slow-model")

    assert len(slept) >= 1
    assert slept[0] > 0


# ---------------------------------------------------------------------------
# set_limit override
# ---------------------------------------------------------------------------

def test_set_limit_overrides_default():
    rate_limiter.set_limit("gemini-2.0-flash", 100)
    assert rate_limiter._LIMITS["gemini-2.0-flash"] == 100
    # Restore so other tests aren't affected
    rate_limiter.set_limit("gemini-2.0-flash", 15)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reset_clears_single_model():
    rate_limiter.set_limit("m1", 10)
    rate_limiter.set_limit("m2", 10)
    await rate_limiter.acquire("m1")
    await rate_limiter.acquire("m2")

    rate_limiter.reset("m1")

    assert len(rate_limiter._windows.get("m1", [])) == 0
    assert len(rate_limiter._windows.get("m2", [])) == 1


@pytest.mark.asyncio
async def test_reset_all_clears_everything():
    rate_limiter.set_limit("m1", 10)
    rate_limiter.set_limit("m2", 10)
    await rate_limiter.acquire("m1")
    await rate_limiter.acquire("m2")

    rate_limiter.reset()

    assert rate_limiter._windows == {}


# ---------------------------------------------------------------------------
# Parallel callers don't deadlock
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parallel_acquires_no_deadlock():
    rate_limiter.set_limit("parallel-model", 20)
    # Fire 10 coroutines simultaneously — all should complete without deadlock
    await asyncio.gather(*[rate_limiter.acquire("parallel-model") for _ in range(10)])
    assert len(rate_limiter._windows["parallel-model"]) == 10


# ---------------------------------------------------------------------------
# Default RPM fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_model_uses_default_limit():
    # Should not raise — uses _DEFAULT_RPM = 15
    await rate_limiter.acquire("totally-unknown-model-xyz")
    assert len(rate_limiter._windows["totally-unknown-model-xyz"]) == 1
