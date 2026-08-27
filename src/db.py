"""SQLite setup and helpers for Phase 0 memory storage.

When DATABASE_URL is set in the environment, the PostgreSQL + pgvector
backend (src/db_postgres.py) overrides all public functions at the bottom
of this module.  The rest of the codebase always imports from ``src.db``
and never needs to know which backend is active.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import sqlite3
import struct
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger("lens.db")

DB_PATH = Path("lens_memory.db")
MEMORY_TTL_DAYS = 30


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS interactions (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                subject_name TEXT NOT NULL,
                location_slug TEXT,
                summary     TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                created_at  TEXT NOT NULL,
                expires_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id);
            CREATE INDEX IF NOT EXISTS idx_interactions_expires ON interactions(expires_at);
            CREATE TABLE IF NOT EXISTS cost_log (
                id          TEXT PRIMARY KEY,
                agent       TEXT NOT NULL,
                model       TEXT NOT NULL,
                input_tokens  INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd    REAL NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_interests (
                user_id     TEXT NOT NULL,
                interest    TEXT NOT NULL,
                score       REAL NOT NULL DEFAULT 1.0,
                updated_at  TEXT NOT NULL,
                PRIMARY KEY (user_id, interest)
            );
            CREATE TABLE IF NOT EXISTS entity_facts (
                id          TEXT PRIMARY KEY,
                entity_name TEXT NOT NULL,
                fact_key    TEXT NOT NULL,
                fact_value  TEXT NOT NULL,
                source      TEXT,
                as_of       TEXT,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_facts_entity ON entity_facts(entity_name);
        """)
        conn.commit()
        logger.info("DB initialised at %s", path)
    finally:
        conn.close()


def _embed_to_blob(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _blob_to_embed(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_score(
    query_terms: list[str],
    doc_text: str,
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Simplified single-document BM25 score — no corpus IDF (uniform weight).

    Works well for short subject_name strings where IDF is less meaningful than
    term frequency relative to document length.
    """
    tokens = _tokenise(doc_text)
    dl = len(tokens) or 1
    tf_map: dict[str, int] = {}
    for t in tokens:
        tf_map[t] = tf_map.get(t, 0) + 1

    score = 0.0
    for term in query_terms:
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        idf = math.log(2.0)  # uniform IDF (single-corpus approximation)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += idf * numerator / denominator
    return score


def _reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Combine multiple ranked lists of indices using RRF.

    Returns list of (original_index, rrf_score) sorted descending.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def write_interaction(
    user_id: str,
    subject_name: str,
    summary: str,
    embedding: list[float],
    location_slug: str | None = None,
) -> str:
    def _write() -> str:
        interaction_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        expires = now + timedelta(days=MEMORY_TTL_DAYS)
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT INTO interactions (id, user_id, subject_name, location_slug, summary, embedding, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    interaction_id,
                    user_id,
                    subject_name,
                    location_slug,
                    summary,
                    _embed_to_blob(embedding),
                    now.isoformat(),
                    expires.isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return interaction_id

    return await asyncio.get_running_loop().run_in_executor(None, _write)


async def search_interactions(
    user_id: str,
    query_embedding: list[float],
    query_text: str = "",
    top_k: int = 5,
) -> list[dict]:
    """Hybrid Layer-2 retrieval: BM25 keyword + semantic cosine, fused via RRF.

    ``query_text`` is the entity name string used for BM25 term matching.
    Falls back to pure semantic when ``query_text`` is empty.
    """

    def _search() -> list[dict]:
        conn = _get_conn()
        try:
            now = datetime.now(UTC).isoformat()
            rows = conn.execute(
                "SELECT id, subject_name, summary, created_at, embedding FROM interactions "
                "WHERE user_id = ? AND expires_at > ?",
                (user_id, now),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        docs = [
            {
                "interaction_id": row["id"],
                "subject_name": row["subject_name"],
                "summary": row["summary"],
                "timestamp": datetime.fromisoformat(row["created_at"]),
                "embedding": _blob_to_embed(row["embedding"]),
            }
            for row in rows
        ]

        # Semantic ranking
        sem_scores = [_cosine_similarity(query_embedding, d["embedding"]) for d in docs]
        sem_ranked = sorted(range(len(docs)), key=lambda i: sem_scores[i], reverse=True)

        ranked_lists = [sem_ranked]

        # BM25 ranking (only when query_text is available)
        if query_text:
            query_terms = _tokenise(query_text)
            avg_dl = max(1, sum(len(_tokenise(d["subject_name"])) for d in docs) / len(docs))
            bm25_scores = [_bm25_score(query_terms, d["subject_name"], avg_dl) for d in docs]
            bm25_ranked = sorted(range(len(docs)), key=lambda i: bm25_scores[i], reverse=True)
            ranked_lists.append(bm25_ranked)

        fused = _reciprocal_rank_fusion(ranked_lists)

        result = []
        for idx, rrf_score in fused[:top_k]:
            d = docs[idx]
            result.append(
                {
                    "interaction_id": d["interaction_id"],
                    "subject_name": d["subject_name"],
                    "summary": d["summary"],
                    "timestamp": d["timestamp"],
                    "similarity_score": sem_scores[idx],  # keep semantic score for threshold
                    "rrf_score": rrf_score,
                }
            )
        return result

    return await asyncio.get_running_loop().run_in_executor(None, _search)


async def upsert_interest(user_id: str, interest: str, weight: float = 1.0) -> None:
    """Increment an interest score; decay existing score by 0.9 to favour recency."""

    def _upsert() -> None:
        conn = _get_conn()
        try:
            existing = conn.execute(
                "SELECT score FROM user_interests WHERE user_id = ? AND interest = ?",
                (user_id, interest),
            ).fetchone()
            new_score = (existing["score"] * 0.9 + weight) if existing else weight
            conn.execute(
                "INSERT INTO user_interests (user_id, interest, score, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, interest) DO UPDATE SET score = excluded.score, updated_at = excluded.updated_at",
                (user_id, interest, new_score, datetime.now(UTC).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    await asyncio.get_running_loop().run_in_executor(None, _upsert)


async def get_user_interests(user_id: str, top_k: int = 10) -> dict[str, float]:
    """Return top-k interests sorted by score descending."""

    def _get() -> dict[str, float]:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT interest, score FROM user_interests WHERE user_id = ? "
                "ORDER BY score DESC LIMIT ?",
                (user_id, top_k),
            ).fetchall()
            return {r["interest"]: round(r["score"], 3) for r in rows}
        finally:
            conn.close()

    return await asyncio.get_running_loop().run_in_executor(None, _get)


async def write_entity_facts(entity_name: str, facts: list[dict]) -> None:
    """Persist search facts for an entity (Layer 3).

    Each fact dict must have ``fact_key`` and ``fact_value`` at minimum.
    Existing facts for this entity are kept — new ones are appended as new
    versions (no overwrite). Caller passes historical_facts + live_facts from
    the SearchResult.
    """

    def _write() -> None:
        now = datetime.now(UTC).isoformat()
        conn = _get_conn()
        try:
            for fact in facts:
                conn.execute(
                    "INSERT INTO entity_facts (id, entity_name, fact_key, fact_value, source, as_of, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        entity_name,
                        str(fact.get("fact_key", "fact")),
                        str(fact.get("fact_value", fact.get("fact", ""))),
                        fact.get("source", ""),
                        fact.get("as_of", None),
                        now,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    await asyncio.get_running_loop().run_in_executor(None, _write)


async def get_entity_facts(entity_name: str, top_k: int = 10) -> list[dict]:
    """Return the most recently written facts for an entity (Layer 3)."""

    def _get() -> list[dict]:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT fact_key, fact_value, source, as_of, created_at FROM entity_facts "
                "WHERE entity_name = ? ORDER BY created_at DESC LIMIT ?",
                (entity_name, top_k),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    return await asyncio.get_running_loop().run_in_executor(None, _get)


async def write_cost_entry(
    agent: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    def _write() -> None:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT INTO cost_log (id, agent, model, input_tokens, output_tokens, cost_usd, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    agent,
                    model,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    await asyncio.get_running_loop().run_in_executor(None, _write)


# ---------------------------------------------------------------------------
# Backend override — PostgreSQL replaces SQLite when DATABASE_URL is set
# ---------------------------------------------------------------------------

if os.environ.get("DATABASE_URL"):
    from src.db_postgres import (  # noqa: F401
        get_entity_facts,
        get_user_interests,
        search_interactions,
        upsert_interest,
        write_cost_entry,
        write_entity_facts,
        write_interaction,
    )
    from src.db_postgres import init_db as init_db  # type: ignore[assignment]  # noqa: F401

    logger.info("PostgreSQL backend active (DATABASE_URL detected)")
