"""PostgreSQL + pgvector backend — Phase 1 drop-in for src/db.py.

Exposes the same async public API as db.py so the rest of the codebase
needs no changes. Activated when DATABASE_URL is set in the environment.

Key differences from the SQLite backend:
- Embeddings stored as vector(3072) — native pgvector type
- Cosine similarity computed by the database: <=> operator (cosine distance)
- BM25 keyword search uses PostgreSQL full-text ts_rank_cd
- No Python-side cosine loop — DB does the heavy lifting
- Concurrent writes safe (asyncpg connection pool)

Schema notes:
- interactions.embedding  vector(3072) — gemini-embedding-001 outputs 3072 dims
- No struct.pack blobs — embeddings stored as pgvector literals
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import UTC, datetime, timedelta

import asyncpg

logger = logging.getLogger("lens.db_postgres")

MEMORY_TTL_DAYS = 30
_pool: asyncpg.Pool | None = None

# gemini-embedding-001 produces 3072-dimensional vectors
_EMBED_DIM = 3072


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        url = os.environ["DATABASE_URL"]

        async def _init(conn: asyncpg.Connection) -> None:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Register pgvector codec so Python lists round-trip cleanly
            await conn.set_type_codec(
                "vector",
                encoder=lambda v: "[" + ",".join(str(x) for x in v) + "]",
                decoder=lambda s: [float(x) for x in s.strip("[]").split(",")],
                schema="public",
                format="text",
            )

        _pool = await asyncpg.create_pool(url, min_size=2, max_size=10, init=_init)
    return _pool


async def init_db() -> None:
    """Create tables and indexes. Safe to call on every startup (idempotent)."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS interactions (
                id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id      TEXT NOT NULL,
                subject_name TEXT NOT NULL,
                location_slug TEXT,
                summary      TEXT NOT NULL,
                embedding    vector({_EMBED_DIM}),
                tsvec        tsvector GENERATED ALWAYS AS (to_tsvector('english', subject_name || ' ' || summary)) STORED,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                expires_at   TIMESTAMPTZ NOT NULL
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_interactions_expires ON interactions(expires_at)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_interactions_tsvec ON interactions USING GIN(tsvec)
        """)
        # HNSW index for fast approximate nearest-neighbour (cosine)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_interactions_hnsw
            ON interactions USING hnsw (embedding vector_cosine_ops)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS cost_log (
                id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                agent         TEXT NOT NULL,
                model         TEXT NOT NULL,
                input_tokens  INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd      REAL NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_interests (
                user_id   TEXT NOT NULL,
                interest  TEXT NOT NULL,
                score     REAL NOT NULL DEFAULT 1.0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (user_id, interest)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_facts (
                id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity_name  TEXT NOT NULL,
                fact_key     TEXT NOT NULL,
                fact_value   TEXT NOT NULL,
                source       TEXT,
                as_of        TEXT,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_entity ON entity_facts(entity_name)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key  TEXT PRIMARY KEY,
                card_json  TEXT NOT NULL,
                entity_json TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                expires_at TIMESTAMPTZ NOT NULL
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_expires ON response_cache(expires_at)
        """)
    logger.info("PostgreSQL DB initialised")


async def write_interaction(
    user_id: str,
    subject_name: str,
    summary: str,
    embedding: list[float],
    location_slug: str | None = None,
) -> str:
    pool = await _get_pool()
    interaction_id = str(uuid.uuid4())
    expires = datetime.now(UTC) + timedelta(days=MEMORY_TTL_DAYS)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO interactions (id, user_id, subject_name, location_slug, summary, embedding, expires_at) "
            "VALUES ($1, $2, $3, $4, $5, $6::vector, $7)",
            interaction_id,
            user_id,
            subject_name,
            location_slug,
            summary,
            "[" + ",".join(str(x) for x in embedding) + "]",
            expires,
        )
    return interaction_id


async def search_interactions(
    user_id: str,
    query_embedding: list[float],
    query_text: str = "",
    top_k: int = 5,
) -> list[dict]:
    """Hybrid retrieval: pgvector ANN cosine + PostgreSQL full-text, fused via RRF."""
    pool = await _get_pool()
    now = datetime.now(UTC)
    vec_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    async with pool.acquire() as conn:
        # Semantic: top-k by cosine similarity (1 - cosine_distance)
        sem_rows = await conn.fetch(
            """
            SELECT id, subject_name, summary, created_at,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM interactions
            WHERE user_id = $2 AND expires_at > $3
            ORDER BY embedding <=> $1::vector
            LIMIT $4
            """,
            vec_literal,
            user_id,
            now,
            top_k * 2,
        )

        # Full-text: ts_rank_cd when query_text provided
        ft_rows: list[asyncpg.Record] = []
        if query_text:
            tsq = " & ".join(re.findall(r"[a-z0-9]+", query_text.lower()))
            if tsq:
                ft_rows = await conn.fetch(
                    """
                    SELECT id, ts_rank_cd(tsvec, to_tsquery('english', $1)) AS ft_score
                    FROM interactions
                    WHERE user_id = $2 AND expires_at > $3
                      AND tsvec @@ to_tsquery('english', $1)
                    ORDER BY ft_score DESC
                    LIMIT $4
                    """,
                    tsq,
                    user_id,
                    now,
                    top_k * 2,
                )

    # Build RRF fusion
    sem_ids = [r["id"] for r in sem_rows]
    ft_ids = [r["id"] for r in ft_rows]
    sem_scores = {r["id"]: float(r["similarity"]) for r in sem_rows}

    all_ids = list(dict.fromkeys(sem_ids + ft_ids))
    rrf: dict[str, float] = {}
    k = 60
    for rank, rid in enumerate(sem_ids):
        rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (k + rank + 1)
    for rank, rid in enumerate(ft_ids):
        rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (k + rank + 1)

    fused_ids = sorted(all_ids, key=lambda i: rrf.get(i, 0.0), reverse=True)[:top_k]

    # Fetch full rows for fused IDs
    id_to_row = {r["id"]: r for r in sem_rows}
    result = []
    for rid in fused_ids:
        row = id_to_row.get(rid)
        if row is None:
            continue
        result.append(
            {
                "interaction_id": str(row["id"]),
                "subject_name": row["subject_name"],
                "summary": row["summary"],
                "timestamp": row["created_at"],
                "similarity_score": sem_scores.get(rid, 0.0),
                "rrf_score": rrf.get(rid, 0.0),
            }
        )
    return result


async def upsert_interest(user_id: str, interest: str, weight: float = 1.0) -> None:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_interests (user_id, interest, score, updated_at)
            VALUES ($1, $2, $3, now())
            ON CONFLICT (user_id, interest) DO UPDATE
                SET score = user_interests.score * 0.9 + $3,
                    updated_at = now()
            """,
            user_id,
            interest,
            weight,
        )


async def get_user_interests(user_id: str, top_k: int = 10) -> dict[str, float]:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT interest, score FROM user_interests WHERE user_id = $1 "
            "ORDER BY score DESC LIMIT $2",
            user_id,
            top_k,
        )
    return {r["interest"]: round(float(r["score"]), 3) for r in rows}


async def write_entity_facts(entity_name: str, facts: list[dict]) -> None:
    if not facts:
        return
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO entity_facts (entity_name, fact_key, fact_value, source, as_of) "
            "VALUES ($1, $2, $3, $4, $5)",
            [
                (
                    entity_name,
                    str(f.get("fact_key", "fact")),
                    str(f.get("fact_value", f.get("fact", ""))),
                    f.get("source", ""),
                    f.get("as_of"),
                )
                for f in facts
            ],
        )


async def get_entity_facts(entity_name: str, top_k: int = 10) -> list[dict]:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT fact_key, fact_value, source, as_of, created_at FROM entity_facts "
            "WHERE entity_name = $1 ORDER BY created_at DESC LIMIT $2",
            entity_name,
            top_k,
        )
    return [dict(r) for r in rows]


async def write_cost_entry(
    agent: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO cost_log (agent, model, input_tokens, output_tokens, cost_usd) "
            "VALUES ($1, $2, $3, $4, $5)",
            agent,
            model,
            input_tokens,
            output_tokens,
            cost_usd,
        )


async def close() -> None:
    """Gracefully close the connection pool. Call on server shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
