"""Session store — scan context available for voice AI follow-up.

Primary: in-memory dict (default). When REDIS_URL is set, sessions are
persisted to Redis and survive VisualOS restarts, which means VoiceOS
voice follow-up calls work correctly even after a server restart.

Sessions expire after SESSION_TTL_HOURS (default 1 hour). The voice repo
fetches a session via GET /session/{session_id} to ground its multi-turn
conversation in the original scan output.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime, timedelta

from src.contracts import HistoricalFact, LiveFact, ScanContext

logger = logging.getLogger("lens.session_store")

SESSION_TTL_HOURS = 1
_SESSION_TTL_SECS = SESSION_TTL_HOURS * 3600
_REDIS_PREFIX = "lens:session:"

_store: dict[str, ScanContext] = {}

# ---------------------------------------------------------------------------
# Optional Redis backend
# ---------------------------------------------------------------------------

_redis: object | None = None  # redis.asyncio.Redis when REDIS_URL is set


def _get_redis() -> object | None:
    global _redis
    if _redis is not None:
        return _redis
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis.asyncio as aioredis
        _redis = aioredis.from_url(redis_url, decode_responses=True)
        logger.info("Session store: Redis backend active at %s", redis_url)
    except ImportError:
        logger.warning("REDIS_URL is set but redis package is not installed — using in-memory store")
    return _redis


async def _redis_set(session: ScanContext) -> None:
    r = _get_redis()
    if r is None:
        return
    key = _REDIS_PREFIX + session.session_id
    payload = session.model_dump_json()
    await r.set(key, payload, ex=_SESSION_TTL_SECS)  # type: ignore[union-attr]


async def _redis_get(session_id: str) -> ScanContext | None:
    r = _get_redis()
    if r is None:
        return None
    key = _REDIS_PREFIX + session_id
    raw = await r.get(key)  # type: ignore[union-attr]
    if raw is None:
        return None
    try:
        return ScanContext.model_validate_json(raw)
    except Exception:
        return None


async def _redis_scan_user(user_id: str) -> list[ScanContext]:
    r = _get_redis()
    if r is None:
        return []
    pattern = _REDIS_PREFIX + "*"
    results: list[ScanContext] = []
    now = datetime.now(UTC)
    async for key in r.scan_iter(pattern):  # type: ignore[union-attr]
        raw = await r.get(key)  # type: ignore[union-attr]
        if raw is None:
            continue
        try:
            ctx = ScanContext.model_validate_json(raw)
            if ctx.user_id == user_id and now <= ctx.expires_at:
                results.append(ctx)
        except Exception:
            pass
    results.sort(key=lambda c: c.scanned_at, reverse=True)
    return results


async def create_session(
    entity_name: str,
    entity_type: str,
    confidence_level: str,
    card_headline: str,
    card_body: str,
    historical_facts: list[HistoricalFact],
    live_facts: list[LiveFact],
    nearby_context: str,
    user_id: str,
    image_b64: str = "",
) -> ScanContext:
    """Create, store, and return a new ScanContext with a fresh session ID."""
    now = datetime.now(UTC)
    session = ScanContext(
        session_id=str(uuid.uuid4()),
        entity_name=entity_name,
        entity_type=entity_type,
        confidence_level=confidence_level,
        card_headline=card_headline,
        card_body=card_body,
        historical_facts=historical_facts,
        live_facts=live_facts,
        nearby_context=nearby_context,
        user_id=user_id,
        scanned_at=now,
        expires_at=now + timedelta(hours=SESSION_TTL_HOURS),
        image_b64=image_b64,
    )
    _store[session.session_id] = session
    await _redis_set(session)
    return session


async def get_session(session_id: str) -> ScanContext | None:
    """Return the session if it exists and has not expired (checks Redis first)."""
    ctx = _store.get(session_id)
    if ctx is None:
        ctx = await _redis_get(session_id)
        if ctx is not None:
            _store[session_id] = ctx  # warm local cache
    if ctx is None:
        return None
    if datetime.now(UTC) > ctx.expires_at:
        _store.pop(session_id, None)
        return None
    return ctx


async def list_sessions(user_id: str, limit: int = 20) -> list[ScanContext]:
    """Return up to `limit` non-expired sessions for a user, newest first."""
    now = datetime.now(UTC)
    # Merge local cache with Redis (Redis may have sessions from other workers)
    local = [ctx for ctx in _store.values() if ctx.user_id == user_id and now <= ctx.expires_at]
    redis_sessions = await _redis_scan_user(user_id)
    seen = {ctx.session_id for ctx in local}
    merged = local + [ctx for ctx in redis_sessions if ctx.session_id not in seen]
    merged.sort(key=lambda c: c.scanned_at, reverse=True)
    return merged[:limit]


def clear_expired() -> int:
    """Remove all expired in-memory sessions. Returns the count removed."""
    now = datetime.now(UTC)
    expired = [sid for sid, ctx in _store.items() if now > ctx.expires_at]
    for sid in expired:
        _store.pop(sid, None)
    return len(expired)


def _clear_all() -> None:
    """Test helper — wipe the entire in-memory store."""
    _store.clear()
