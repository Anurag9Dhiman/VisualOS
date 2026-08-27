"""Layer 1 memory — in-context rolling window of recent scans.

Lightweight in-memory store. Survives for the server process lifetime (~1h).
No DB, no embeddings — just a deque of the last N scan summaries per user.

Injected into MemoryResult as ``recent_context`` so the Reasoning and Fusion
steps know what the user looked at recently without touching the DB.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

_MAX_ENTRIES = 5
_TTL_S = 3600  # 1 hour per entry


@dataclass
class _ScanEntry:
    entity_name: str
    summary: str
    scanned_at: float = field(default_factory=time.monotonic)


_store: dict[str, deque[_ScanEntry]] = {}


def record_scan(user_id: str, entity_name: str, summary: str) -> None:
    """Record a completed scan. Called fire-and-forget from write_memory_node."""
    if user_id not in _store:
        _store[user_id] = deque(maxlen=_MAX_ENTRIES)
    _store[user_id].append(_ScanEntry(entity_name=entity_name, summary=summary))


def get_recent_context(user_id: str) -> str:
    """Return a compact sentence about what the user scanned recently.

    Entries older than 1 hour are silently dropped.
    Returns an empty string for new users or after the TTL expires.
    """
    entries = _store.get(user_id)
    if not entries:
        return ""

    now = time.monotonic()
    recent = [e for e in entries if now - e.scanned_at < _TTL_S]
    if not recent:
        return ""

    names = [e.entity_name for e in recent[-3:]]
    if len(names) == 1:
        return f"User recently scanned: {names[0]}."
    joined = ", ".join(names[:-1])
    return f"User recently scanned: {joined} and {names[-1]}."


def clear(user_id: str | None = None) -> None:
    """Reset L1 state. Used in tests."""
    if user_id is None:
        _store.clear()
    elif user_id in _store:
        del _store[user_id]
