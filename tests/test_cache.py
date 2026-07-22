"""Tests for the response cache — no API calls needed.

Covers: make_cache_key determinism, cache_get miss/hit/expiry,
cache_set round-trip, TTL enforcement, and not-initialised guard.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src import cache as cache_mod

# ---------------------------------------------------------------------------
# Fixture: initialise cache against the tmp_db from conftest
# ---------------------------------------------------------------------------

@pytest.fixture()
def cache(tmp_db: Path):
    """Initialise the cache module against the temp DB for each test."""
    cache_mod.init_cache(tmp_db)
    yield
    # Reset global state so tests don't bleed into each other
    cache_mod._DB_PATH = None


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

def test_cache_key_is_deterministic():
    img = b"fake-image-bytes"
    k1 = cache_mod.make_cache_key(img, 12.95, 77.58)
    k2 = cache_mod.make_cache_key(img, 12.95, 77.58)
    assert k1 == k2


def test_cache_key_is_hex_64_chars():
    key = cache_mod.make_cache_key(b"img", 0.0, 0.0)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_cache_key_differs_for_different_images():
    k1 = cache_mod.make_cache_key(b"image-a", 12.0, 77.0)
    k2 = cache_mod.make_cache_key(b"image-b", 12.0, 77.0)
    assert k1 != k2


def test_cache_key_differs_for_different_locations():
    img = b"same-image"
    k1 = cache_mod.make_cache_key(img, 12.950, 77.580)
    k2 = cache_mod.make_cache_key(img, 12.951, 77.580)
    assert k1 != k2


def test_cache_key_rounds_location_to_3_decimals():
    img = b"same-image"
    # 12.9501 and 12.9504 both round to 12.950 → same key
    k1 = cache_mod.make_cache_key(img, 12.9501, 77.5801)
    k2 = cache_mod.make_cache_key(img, 12.9504, 77.5804)
    assert k1 == k2


def test_cache_key_handles_none_location():
    k1 = cache_mod.make_cache_key(b"img", None, None)
    k2 = cache_mod.make_cache_key(b"img", 0.0, 0.0)
    assert k1 == k2  # None is treated as 0.0


# ---------------------------------------------------------------------------
# cache_get / cache_set round-trip
# ---------------------------------------------------------------------------

_SAMPLE_CARD = {
    "card_type": "normal",
    "headline": "The Eiffel Tower",
    "body": "Built 1887–1889.",
    "personalized_hooks": [],
    "citations": [],
    "confidence_displayed": "high",
    "source_mix": {"used_vision": True, "used_memory": False, "used_search": True},
    "cost_usd_total": 0.001,
    "latency_ms": 450,
}


@pytest.mark.asyncio
async def test_cache_get_miss(cache):
    result = await cache_mod.cache_get("nonexistent-key")
    assert result is None


@pytest.mark.asyncio
async def test_cache_set_then_get_hit(cache):
    key = "test-key-abc"
    await cache_mod.cache_set(key, _SAMPLE_CARD)
    result = await cache_mod.cache_get(key)
    assert result is not None
    assert result["headline"] == "The Eiffel Tower"
    assert result["card_type"] == "normal"


@pytest.mark.asyncio
async def test_cache_set_overwrites_existing(cache):
    key = "overwrite-key"
    await cache_mod.cache_set(key, {"headline": "v1"})
    await cache_mod.cache_set(key, {"headline": "v2"})
    result = await cache_mod.cache_get(key)
    assert result["headline"] == "v2"


@pytest.mark.asyncio
async def test_cache_stores_full_card_structure(cache):
    key = "full-card-key"
    await cache_mod.cache_set(key, _SAMPLE_CARD)
    result = await cache_mod.cache_get(key)
    assert result == _SAMPLE_CARD


# ---------------------------------------------------------------------------
# TTL enforcement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_get_returns_none_for_expired_entry(cache, tmp_db: Path):
    key = "expired-key"
    past = (datetime.utcnow() - timedelta(hours=1)).isoformat()

    conn = sqlite3.connect(tmp_db)
    conn.execute(
        "INSERT INTO response_cache (cache_key, card_json, created_at, expires_at) "
        "VALUES (?, ?, ?, ?)",
        (key, json.dumps({"headline": "old"}), past, past),
    )
    conn.commit()
    conn.close()

    result = await cache_mod.cache_get(key)
    assert result is None


@pytest.mark.asyncio
async def test_cache_ttl_is_24_hours(cache, tmp_db: Path):
    key = "ttl-check-key"
    await cache_mod.cache_set(key, {"x": 1})

    conn = sqlite3.connect(tmp_db)
    row = conn.execute(
        "SELECT created_at, expires_at FROM response_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    conn.close()

    created = datetime.fromisoformat(row[0])
    expires = datetime.fromisoformat(row[1])
    assert (expires - created).total_seconds() == pytest.approx(24 * 3600, abs=5)


# ---------------------------------------------------------------------------
# Not-initialised guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_get_raises_if_not_initialised():
    cache_mod._DB_PATH = None
    with pytest.raises(RuntimeError, match="not initialised"):
        await cache_mod.cache_get("any-key")


@pytest.mark.asyncio
async def test_cache_set_raises_if_not_initialised():
    cache_mod._DB_PATH = None
    with pytest.raises(RuntimeError, match="not initialised"):
        await cache_mod.cache_set("any-key", {"x": 1})
