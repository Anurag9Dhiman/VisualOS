"""Tests for the scan session store — create, get, TTL expiry, clear."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.contracts import HistoricalFact, LiveFact
from src.session_store import (
    SESSION_TTL_HOURS,
    _clear_all,
    clear_expired,
    create_session,
    get_session,
    list_sessions,
)

# ---------------------------------------------------------------------------
# Fixture — wipe store before every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_store():
    _clear_all()
    yield
    _clear_all()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_session(**overrides):
    defaults = {
        "entity_name": "Lalbagh Gate",
        "entity_type": "monument",
        "confidence_level": "certain",
        "card_headline": "Lalbagh Botanical Garden — West Gate",
        "card_body": "A colonial-era gate in Bangalore.",
        "historical_facts": [HistoricalFact(fact="Built in 1760.", source="wikipedia")],
        "live_facts": [],
        "nearby_context": "Inside Lalbagh Botanical Garden.",
        "user_id": "user-42",
    }
    defaults.update(overrides)
    return await create_session(**defaults)


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


async def test_create_session_returns_scan_context():
    ctx = await _make_session()
    assert ctx.entity_name == "Lalbagh Gate"
    assert ctx.entity_type == "monument"
    assert ctx.confidence_level == "certain"


async def test_create_session_assigns_uuid():
    ctx = await _make_session()
    assert len(ctx.session_id) == 36
    assert ctx.session_id.count("-") == 4


async def test_create_session_sets_ttl():
    ctx = await _make_session()
    expected_expiry = ctx.scanned_at + timedelta(hours=SESSION_TTL_HOURS)
    assert abs((ctx.expires_at - expected_expiry).total_seconds()) < 1


async def test_create_session_stores_historical_facts():
    ctx = await _make_session(
        historical_facts=[
            HistoricalFact(fact="Founded 1760.", source="wikipedia"),
            HistoricalFact(fact="Named after Hyder Ali.", source="wikidata"),
        ]
    )
    assert len(ctx.historical_facts) == 2
    assert ctx.historical_facts[0].fact == "Founded 1760."


async def test_create_session_stores_live_facts():
    ctx = await _make_session(
        live_facts=[LiveFact(fact="Open 6am–7pm.", source="tavily", as_of="2026-08-11")]
    )
    assert len(ctx.live_facts) == 1
    assert ctx.live_facts[0].as_of == "2026-08-11"


async def test_two_sessions_have_different_ids():
    ctx1 = await _make_session()
    ctx2 = await _make_session()
    assert ctx1.session_id != ctx2.session_id


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


async def test_get_session_returns_stored_context():
    ctx = await _make_session()
    fetched = await get_session(ctx.session_id)
    assert fetched is not None
    assert fetched.session_id == ctx.session_id
    assert fetched.entity_name == "Lalbagh Gate"


async def test_get_session_unknown_id_returns_none():
    assert await get_session("00000000-0000-0000-0000-000000000000") is None


async def test_get_session_expired_returns_none(monkeypatch):
    ctx = await _make_session()
    from src import session_store

    expired_ctx = ctx.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)})
    session_store._store[ctx.session_id] = expired_ctx
    assert await get_session(ctx.session_id) is None


async def test_get_session_expired_removes_from_store(monkeypatch):
    ctx = await _make_session()
    from src import session_store

    session_store._store[ctx.session_id] = ctx.model_copy(
        update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    await get_session(ctx.session_id)
    assert ctx.session_id not in session_store._store


# ---------------------------------------------------------------------------
# clear_expired
# ---------------------------------------------------------------------------


async def test_clear_expired_removes_only_expired():
    active = await _make_session()
    expired = await _make_session(entity_name="Old Scan")

    from src import session_store

    session_store._store[expired.session_id] = expired.model_copy(
        update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
    )

    removed = clear_expired()

    assert removed == 1
    assert await get_session(active.session_id) is not None
    assert await get_session(expired.session_id) is None


async def test_clear_expired_returns_zero_when_all_active():
    await _make_session()
    await _make_session()
    assert clear_expired() == 0


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------


async def test_list_sessions_returns_user_sessions():
    await _make_session(user_id="alice")
    await _make_session(user_id="alice")
    await _make_session(user_id="bob")
    results = await list_sessions("alice")
    assert len(results) == 2
    assert all(s.user_id == "alice" for s in results)


async def test_list_sessions_newest_first():
    import time

    s1 = await _make_session(entity_name="First")
    time.sleep(0.01)
    s2 = await _make_session(entity_name="Second")
    results = await list_sessions("user-42")
    assert results[0].session_id == s2.session_id
    assert results[1].session_id == s1.session_id


async def test_list_sessions_excludes_expired():
    active = await _make_session()
    expired = await _make_session(entity_name="Old")

    from src import session_store

    session_store._store[expired.session_id] = expired.model_copy(
        update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    results = await list_sessions("user-42")
    assert len(results) == 1
    assert results[0].session_id == active.session_id


async def test_list_sessions_respects_limit():
    for i in range(5):
        await _make_session(entity_name=f"Scan {i}")
    results = await list_sessions("user-42", limit=3)
    assert len(results) == 3


async def test_list_sessions_empty_for_unknown_user():
    await _make_session(user_id="alice")
    assert await list_sessions("nobody") == []
