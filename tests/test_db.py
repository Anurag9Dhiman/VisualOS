"""Tests for SQLite db helpers — no API calls needed."""

from __future__ import annotations

import struct
from datetime import datetime, timedelta

import pytest

from src import db as db_module


def _make_embedding(dim: int = 8, value: float = 0.5) -> list[float]:
    return [value] * dim


# ---------------------------------------------------------------------------
# write_interaction / search_interactions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_and_search(tmp_db):
    embed = _make_embedding()
    iid = await db_module.write_interaction(
        user_id="u1",
        subject_name="Eiffel Tower",
        summary="Famous iron tower in Paris.",
        embedding=embed,
    )
    assert isinstance(iid, str) and len(iid) == 36  # UUID

    results = await db_module.search_interactions("u1", embed, top_k=5)
    assert len(results) == 1
    assert results[0]["subject_name"] == "Eiffel Tower"
    assert results[0]["similarity_score"] == pytest.approx(1.0, rel=1e-3)


@pytest.mark.asyncio
async def test_search_returns_top_k(tmp_db):
    for i in range(5):
        await db_module.write_interaction(
            user_id="u1",
            subject_name=f"Place {i}",
            summary=f"Summary {i}",
            embedding=_make_embedding(value=i * 0.1 + 0.1),
        )

    results = await db_module.search_interactions(
        "u1", _make_embedding(value=0.5), top_k=3
    )
    assert len(results) == 3


@pytest.mark.asyncio
async def test_search_excludes_other_user(tmp_db):
    embed = _make_embedding()
    await db_module.write_interaction("u1", "Tower", "s", embed)
    results = await db_module.search_interactions("u2", embed, top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_expired_interactions_excluded(tmp_db, monkeypatch):
    # Write an interaction that is already expired
    import sqlite3
    conn = sqlite3.connect(tmp_db)
    past = (datetime.utcnow() - timedelta(days=1)).isoformat()
    embed_blob = struct.pack("8f", *_make_embedding())
    conn.execute(
        "INSERT INTO interactions (id, user_id, subject_name, location_slug, summary, embedding, created_at, expires_at) "
        "VALUES ('expired-id', 'u1', 'Old Place', NULL, 'Old summary', ?, ?, ?)",
        (embed_blob, past, past),
    )
    conn.commit()
    conn.close()

    results = await db_module.search_interactions("u1", _make_embedding(), top_k=5)
    assert all(r["subject_name"] != "Old Place" for r in results)


# ---------------------------------------------------------------------------
# Embedding round-trip
# ---------------------------------------------------------------------------

def test_embed_blob_roundtrip():
    original = [0.1, 0.2, 0.3, -0.5, 1.0]
    blob = db_module._embed_to_blob(original)
    recovered = db_module._blob_to_embed(blob)
    assert len(recovered) == len(original)
    for a, b in zip(original, recovered):
        assert a == pytest.approx(b, rel=1e-5)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert db_module._cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert db_module._cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert db_module._cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_zero_vector():
    assert db_module._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# TTL — interactions expire after 30 days
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ttl_is_30_days(tmp_db):
    import sqlite3
    embed = _make_embedding()
    iid = await db_module.write_interaction("u1", "Test", "s", embed)

    conn = sqlite3.connect(tmp_db)
    row = conn.execute(
        "SELECT created_at, expires_at FROM interactions WHERE id = ?", (iid,)
    ).fetchone()
    conn.close()

    created = datetime.fromisoformat(row[0])
    expires = datetime.fromisoformat(row[1])
    delta = expires - created
    assert delta.days == db_module.MEMORY_TTL_DAYS


# ---------------------------------------------------------------------------
# upsert_interest — exponential decay scoring
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_interest_creates_entry(tmp_db):
    await db_module.upsert_interest("u1", "architecture")
    interests = await db_module.get_user_interests("u1")
    assert "architecture" in interests
    assert interests["architecture"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_upsert_interest_decays_on_repeat(tmp_db):
    await db_module.upsert_interest("u1", "history")
    await db_module.upsert_interest("u1", "history")
    interests = await db_module.get_user_interests("u1")
    # Second call: new_score = 1.0 * 0.9 + 1.0 = 1.9
    assert interests["history"] == pytest.approx(1.9)


@pytest.mark.asyncio
async def test_upsert_interest_multiple_users_isolated(tmp_db):
    await db_module.upsert_interest("u1", "botany")
    await db_module.upsert_interest("u2", "sculpture")
    u1 = await db_module.get_user_interests("u1")
    u2 = await db_module.get_user_interests("u2")
    assert "botany" in u1 and "sculpture" not in u1
    assert "sculpture" in u2 and "botany" not in u2


@pytest.mark.asyncio
async def test_upsert_interest_multiple_categories(tmp_db):
    await db_module.upsert_interest("u1", "architecture")
    await db_module.upsert_interest("u1", "history")
    interests = await db_module.get_user_interests("u1")
    assert len(interests) == 2


# ---------------------------------------------------------------------------
# write_cost_entry — persists cost entries to SQLite
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_cost_entry_writes_row(tmp_db):
    import sqlite3
    await db_module.write_cost_entry(
        agent="vision", model="gemini-2.0-flash",
        input_tokens=100, output_tokens=50, cost_usd=0.000038,
    )
    conn = sqlite3.connect(tmp_db)
    row = conn.execute("SELECT agent, model, input_tokens, output_tokens, cost_usd FROM cost_log").fetchone()
    conn.close()
    assert row[0] == "vision"
    assert row[1] == "gemini-2.0-flash"
    assert row[2] == 100
    assert row[3] == 50
    assert row[4] == pytest.approx(0.000038)


@pytest.mark.asyncio
async def test_write_cost_entry_multiple_entries(tmp_db):
    import sqlite3
    await db_module.write_cost_entry("vision", "gemini-2.0-flash", 100, 50, 0.001)
    await db_module.write_cost_entry("search", "gemini-2.0-flash", 200, 80, 0.002)
    conn = sqlite3.connect(tmp_db)
    count = conn.execute("SELECT COUNT(*) FROM cost_log").fetchone()[0]
    conn.close()
    assert count == 2
