"""Tests for SQLite db helpers — no API calls needed."""

from __future__ import annotations

import struct
from datetime import UTC, datetime, timedelta

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

    results = await db_module.search_interactions("u1", _make_embedding(value=0.5), top_k=3)
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
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
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
        agent="vision",
        model="gemini-2.0-flash",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.000038,
    )
    conn = sqlite3.connect(tmp_db)
    row = conn.execute(
        "SELECT agent, model, input_tokens, output_tokens, cost_usd FROM cost_log"
    ).fetchone()
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


# ---------------------------------------------------------------------------
# BM25 helpers — _tokenise, _bm25_score, _reciprocal_rank_fusion
# ---------------------------------------------------------------------------


def test_tokenise_lowercases_and_splits():
    tokens = db_module._tokenise("India Gate 2024")
    assert tokens == ["india", "gate", "2024"]


def test_tokenise_strips_punctuation():
    tokens = db_module._tokenise("Eiffel Tower, Paris!")
    assert "eiffel" in tokens
    assert "tower" in tokens
    assert "paris" in tokens
    assert "," not in tokens


def test_tokenise_empty_string():
    assert db_module._tokenise("") == []


def test_bm25_exact_match_scores_higher_than_no_match():
    query = db_module._tokenise("India Gate")
    score_match = db_module._bm25_score(query, "India Gate monument", avg_dl=3.0)
    score_no_match = db_module._bm25_score(query, "Colosseum amphitheatre Rome", avg_dl=3.0)
    assert score_match > score_no_match


def test_bm25_zero_for_empty_query():
    assert db_module._bm25_score([], "India Gate monument", avg_dl=3.0) == 0.0


def test_bm25_zero_for_no_overlap():
    query = db_module._tokenise("Eiffel Tower")
    score = db_module._bm25_score(query, "Colosseum Rome", avg_dl=2.0)
    assert score == 0.0


def test_rrf_combines_two_rankings():
    # Two identical rankings → same order preserved
    ranked = [0, 1, 2]
    fused = db_module._reciprocal_rank_fusion([ranked, ranked])
    order = [idx for idx, _ in fused]
    assert order == [0, 1, 2]


def test_rrf_combines_conflicting_rankings():
    # Ranking A prefers 0, Ranking B prefers 1 → item appearing in both gets combined score
    fused = db_module._reciprocal_rank_fusion([[0, 1], [1, 0]])
    scores = dict(fused)
    # Both 0 and 1 appear in both lists — item 0 is rank-1 in A and rank-2 in B; item 1 vice versa
    # RRF is symmetric here, so scores should be equal
    assert scores[0] == pytest.approx(scores[1])


def test_rrf_item_only_in_one_list_scores_lower():
    # item 2 only in one list — should score lower than items in both
    fused = db_module._reciprocal_rank_fusion([[0, 1, 2], [0, 1]])
    scores = dict(fused)
    assert scores[0] > scores[2]  # 0 in both lists > 2 in only one


# ---------------------------------------------------------------------------
# Hybrid search — search_interactions with query_text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_search_boosts_exact_name_match(tmp_db):
    """BM25 should boost an exact name match above a semantically similar but differently-named entity."""
    # Same embedding for both — semantic score identical
    embed = _make_embedding(value=0.5)

    await db_module.write_interaction("u1", "India Gate", "War memorial in New Delhi.", embed)
    await db_module.write_interaction(
        "u1", "Rajpath Boulevard", "Road leading to India Gate.", embed
    )

    # With hybrid search, "India Gate" query_text should boost the exact match
    results = await db_module.search_interactions("u1", embed, query_text="India Gate", top_k=2)
    assert len(results) == 2
    # India Gate should be ranked first due to BM25 boost
    assert results[0]["subject_name"] == "India Gate"


@pytest.mark.asyncio
async def test_hybrid_search_fallback_without_query_text(tmp_db):
    """query_text='' falls back to pure semantic — results still returned."""
    embed = _make_embedding(value=0.5)
    await db_module.write_interaction("u1", "Eiffel Tower", "Iron tower in Paris.", embed)

    results = await db_module.search_interactions("u1", embed, query_text="", top_k=5)
    assert len(results) == 1
    assert results[0]["subject_name"] == "Eiffel Tower"


# ---------------------------------------------------------------------------
# Layer 3 — write_entity_facts / get_entity_facts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_and_read_entity_facts(tmp_db):
    facts = [
        {"fact_key": "historical", "fact_value": "Built in 1931", "source": "Wikipedia"},
        {
            "fact_key": "live",
            "fact_value": "Open 24 hours",
            "source": "Travily",
            "as_of": "2026-01",
        },
    ]
    await db_module.write_entity_facts("India Gate", facts)

    results = await db_module.get_entity_facts("India Gate")
    assert len(results) == 2
    values = {r["fact_value"] for r in results}
    assert "Built in 1931" in values
    assert "Open 24 hours" in values


@pytest.mark.asyncio
async def test_entity_facts_different_entities_isolated(tmp_db):
    await db_module.write_entity_facts(
        "India Gate", [{"fact_key": "historical", "fact_value": "Built 1931", "source": "Wiki"}]
    )
    await db_module.write_entity_facts(
        "Colosseum", [{"fact_key": "historical", "fact_value": "Built 70 AD", "source": "Wiki"}]
    )

    india_gate = await db_module.get_entity_facts("India Gate")
    colosseum = await db_module.get_entity_facts("Colosseum")

    assert all(r["fact_value"] != "Built 70 AD" for r in india_gate)
    assert all(r["fact_value"] != "Built 1931" for r in colosseum)


@pytest.mark.asyncio
async def test_entity_facts_empty_for_unknown_entity(tmp_db):
    results = await db_module.get_entity_facts("Unknown Entity XYZ")
    assert results == []


@pytest.mark.asyncio
async def test_entity_facts_top_k_limits_results(tmp_db):
    facts = [{"fact_key": "f", "fact_value": f"fact {i}", "source": "s"} for i in range(15)]
    await db_module.write_entity_facts("Big Entity", facts)

    results = await db_module.get_entity_facts("Big Entity", top_k=5)
    assert len(results) == 5


@pytest.mark.asyncio
async def test_entity_facts_preserves_versions(tmp_db):
    """Writing facts twice for the same entity appends — does not overwrite."""
    fact_v1 = [{"fact_key": "hours", "fact_value": "Open 9-5", "source": "site"}]
    fact_v2 = [{"fact_key": "hours", "fact_value": "Open 8-6", "source": "site"}]

    await db_module.write_entity_facts("Museum", fact_v1)
    await db_module.write_entity_facts("Museum", fact_v2)

    results = await db_module.get_entity_facts("Museum", top_k=10)
    values = {r["fact_value"] for r in results}
    # Both versions preserved
    assert "Open 9-5" in values
    assert "Open 8-6" in values


@pytest.mark.asyncio
async def test_entity_facts_live_fact_as_of_stored(tmp_db):
    facts = [
        {"fact_key": "live", "fact_value": "Closed today", "source": "web", "as_of": "2026-08"}
    ]
    await db_module.write_entity_facts("Gallery", facts)

    results = await db_module.get_entity_facts("Gallery")
    assert results[0]["as_of"] == "2026-08"


@pytest.mark.asyncio
async def test_write_entity_facts_empty_list_is_noop(tmp_db):
    """Writing an empty facts list should not raise and should store nothing."""
    await db_module.write_entity_facts("Empty Entity", [])
    results = await db_module.get_entity_facts("Empty Entity")
    assert results == []
