"""SQLite setup and helpers for Phase 0 memory storage."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import struct
import uuid
from datetime import datetime, timedelta
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
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


async def write_interaction(
    user_id: str,
    subject_name: str,
    summary: str,
    embedding: list[float],
    location_slug: str | None = None,
) -> str:
    def _write() -> str:
        interaction_id = str(uuid.uuid4())
        now = datetime.utcnow()
        expires = now + timedelta(days=MEMORY_TTL_DAYS)
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT INTO interactions (id, user_id, subject_name, location_slug, summary, embedding, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (interaction_id, user_id, subject_name, location_slug, summary,
                 _embed_to_blob(embedding), now.isoformat(), expires.isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
        return interaction_id

    return await asyncio.get_event_loop().run_in_executor(None, _write)


async def search_interactions(
    user_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    def _search() -> list[dict]:
        conn = _get_conn()
        try:
            now = datetime.utcnow().isoformat()
            rows = conn.execute(
                "SELECT id, subject_name, summary, created_at, embedding FROM interactions "
                "WHERE user_id = ? AND expires_at > ?",
                (user_id, now),
            ).fetchall()
        finally:
            conn.close()

        scored = []
        for row in rows:
            stored_embed = _blob_to_embed(row["embedding"])
            score = _cosine_similarity(query_embedding, stored_embed)
            scored.append({
                "interaction_id": row["id"],
                "subject_name": row["subject_name"],
                "summary": row["summary"],
                "timestamp": datetime.fromisoformat(row["created_at"]),
                "similarity_score": score,
            })

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:top_k]

    return await asyncio.get_event_loop().run_in_executor(None, _search)


async def write_cost_entry(
    agent: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float,
) -> None:
    def _write() -> None:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT INTO cost_log (id, agent, model, input_tokens, output_tokens, cost_usd, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), agent, model, input_tokens, output_tokens,
                 cost_usd, datetime.utcnow().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    await asyncio.get_event_loop().run_in_executor(None, _write)
