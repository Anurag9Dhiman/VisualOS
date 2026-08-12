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


def _make_session(**overrides):
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
    return create_session(**defaults)


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


def test_create_session_returns_scan_context():
    ctx = _make_session()
    assert ctx.entity_name == "Lalbagh Gate"
    assert ctx.entity_type == "monument"
    assert ctx.confidence_level == "certain"


def test_create_session_assigns_uuid():
    ctx = _make_session()
    assert len(ctx.session_id) == 36
    assert ctx.session_id.count("-") == 4


def test_create_session_sets_ttl():
    ctx = _make_session()
    expected_expiry = ctx.scanned_at + timedelta(hours=SESSION_TTL_HOURS)
    assert abs((ctx.expires_at - expected_expiry).total_seconds()) < 1


def test_create_session_stores_historical_facts():
    ctx = _make_session(
        historical_facts=[
            HistoricalFact(fact="Founded 1760.", source="wikipedia"),
            HistoricalFact(fact="Named after Hyder Ali.", source="wikidata"),
        ]
    )
    assert len(ctx.historical_facts) == 2
    assert ctx.historical_facts[0].fact == "Founded 1760."


def test_create_session_stores_live_facts():
    ctx = _make_session(
        live_facts=[LiveFact(fact="Open 6am–7pm.", source="tavily", as_of="2026-08-11")]
    )
    assert len(ctx.live_facts) == 1
    assert ctx.live_facts[0].as_of == "2026-08-11"


def test_two_sessions_have_different_ids():
    ctx1 = _make_session()
    ctx2 = _make_session()
    assert ctx1.session_id != ctx2.session_id


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


def test_get_session_returns_stored_context():
    ctx = _make_session()
    fetched = get_session(ctx.session_id)
    assert fetched is not None
    assert fetched.session_id == ctx.session_id
    assert fetched.entity_name == "Lalbagh Gate"


def test_get_session_unknown_id_returns_none():
    assert get_session("00000000-0000-0000-0000-000000000000") is None


def test_get_session_expired_returns_none(monkeypatch):
    ctx = _make_session()
    # Monkey-patch expires_at to the past
    from src import session_store

    expired_ctx = ctx.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)})
    session_store._store[ctx.session_id] = expired_ctx
    assert get_session(ctx.session_id) is None


def test_get_session_expired_removes_from_store(monkeypatch):
    ctx = _make_session()
    from src import session_store

    session_store._store[ctx.session_id] = ctx.model_copy(
        update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    get_session(ctx.session_id)
    assert ctx.session_id not in session_store._store


# ---------------------------------------------------------------------------
# clear_expired
# ---------------------------------------------------------------------------


def test_clear_expired_removes_only_expired():
    active = _make_session()
    expired = _make_session(entity_name="Old Scan")

    from src import session_store

    session_store._store[expired.session_id] = expired.model_copy(
        update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
    )

    removed = clear_expired()

    assert removed == 1
    assert get_session(active.session_id) is not None
    assert get_session(expired.session_id) is None


def test_clear_expired_returns_zero_when_all_active():
    _make_session()
    _make_session()
    assert clear_expired() == 0
