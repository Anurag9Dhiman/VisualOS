"""Response cache — skip all agents on repeated image+location queries.

Cache key: SHA256(preprocessed_image_bytes + rounded_lat + rounded_lng)
TTL: 24 hours — facts can go stale, so we don't cache indefinitely.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("lens.cache")

_TTL_HOURS = 24
_DB_PATH: Path | None = None  # set by init_cache()


def init_cache(db_path: Path) -> None:
    """Create the response_cache table if it doesn't exist."""
    global _DB_PATH
    _DB_PATH = db_path
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key   TEXT PRIMARY KEY,
                card_json   TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                expires_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cache_expires ON response_cache(expires_at);
        """)
        conn.commit()
    finally:
        conn.close()


def make_cache_key(image_bytes: bytes, lat: float | None, lng: float | None) -> str:
    """Stable key from image content + location rounded to ~100m precision."""
    lat_r = round(lat or 0.0, 3)
    lng_r = round(lng or 0.0, 3)
    loc = f"{lat_r},{lng_r}".encode()
    return hashlib.sha256(image_bytes + loc).hexdigest()


async def cache_get(cache_key: str) -> dict | None:
    """Return cached card dict or None on miss/expiry."""
    def _get() -> dict | None:
        if _DB_PATH is None:
            raise RuntimeError("cache not initialised — call init_cache() first")
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            now = datetime.utcnow().isoformat()
            row = conn.execute(
                "SELECT card_json FROM response_cache WHERE cache_key = ? AND expires_at > ?",
                (cache_key, now),
            ).fetchone()
            return json.loads(row["card_json"]) if row else None
        finally:
            conn.close()

    result = await asyncio.get_event_loop().run_in_executor(None, _get)
    if result:
        logger.info("Cache HIT for key %s…", cache_key[:12])
    else:
        logger.debug("Cache MISS for key %s…", cache_key[:12])
    return result


async def cache_set(cache_key: str, card_dict: dict) -> None:
    """Store a card dict with a 24-hour TTL."""
    def _set() -> None:
        if _DB_PATH is None:
            raise RuntimeError("cache not initialised — call init_cache() first")
        now = datetime.utcnow()
        expires = now + timedelta(hours=_TTL_HOURS)
        conn = sqlite3.connect(_DB_PATH)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO response_cache (cache_key, card_json, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (cache_key, json.dumps(card_dict), now.isoformat(), expires.isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    await asyncio.get_event_loop().run_in_executor(None, _set)
    logger.debug("Cache SET for key %s… (TTL %dh)", cache_key[:12], _TTL_HOURS)
