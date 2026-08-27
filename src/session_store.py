"""In-memory session store — scan context available for voice AI follow-up.

Sessions expire after SESSION_TTL_HOURS (default 1 hour). The voice repo
fetches a session via GET /session/{session_id} to ground its multi-turn
conversation in the original scan output.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from src.contracts import HistoricalFact, LiveFact, ScanContext

SESSION_TTL_HOURS = 1

_store: dict[str, ScanContext] = {}


def create_session(
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
    return session


def get_session(session_id: str) -> ScanContext | None:
    """Return the session if it exists and has not expired."""
    ctx = _store.get(session_id)
    if ctx is None:
        return None
    if datetime.now(UTC) > ctx.expires_at:
        _store.pop(session_id, None)
        return None
    return ctx


def list_sessions(user_id: str, limit: int = 20) -> list[ScanContext]:
    """Return up to `limit` non-expired sessions for a user, newest first."""
    now = datetime.now(UTC)
    results = [
        ctx for ctx in _store.values()
        if ctx.user_id == user_id and now <= ctx.expires_at
    ]
    results.sort(key=lambda c: c.scanned_at, reverse=True)
    return results[:limit]


def clear_expired() -> int:
    """Remove all expired sessions. Returns the count removed."""
    now = datetime.now(UTC)
    expired = [sid for sid, ctx in _store.items() if now > ctx.expires_at]
    for sid in expired:
        _store.pop(sid, None)
    return len(expired)


def _clear_all() -> None:
    """Test helper — wipe the entire store."""
    _store.clear()
