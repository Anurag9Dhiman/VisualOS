"""Tests for SQLite db helpers — no API calls needed."""

from __future__ import annotations

import struct
from datetime import datetime, timedelta
from pathlib import Path

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
