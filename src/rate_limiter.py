"""Sliding-window rate limiter for Gemini API calls.

One shared limiter per process. Tracks call timestamps per model in a
60-second window. When the limit is reached, `acquire()` sleeps until a slot
opens — lock is released during sleep so parallel callers don't pile up.

Limits are keyed by model name and default to the Gemini free-tier values.
Override per-model at startup by calling `set_limit(model, rpm)`.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time

logger = logging.getLogger("lens.rate_limiter")

_WINDOW_S = 60.0

# Gemini free-tier defaults: https://ai.google.dev/gemini-api/docs/rate-limits
_LIMITS: dict[str, int] = {
    "gemini-2.0-flash": 15,
    "text-embedding-004": 1500,
}
_DEFAULT_RPM = 15

_windows: dict[str, collections.deque[float]] = {}
_locks: dict[str, asyncio.Lock] = {}


def set_limit(model: str, rpm: int) -> None:
    """Override the default RPM limit for a model. Call before first acquire."""
    _LIMITS[model] = rpm


def _ensure(model: str) -> tuple[collections.deque[float], asyncio.Lock]:
    if model not in _windows:
        _windows[model] = collections.deque()
        _locks[model] = asyncio.Lock()
    return _windows[model], _locks[model]


async def acquire(model: str) -> None:
    """Wait until a request slot is available for `model`.

    Uses a sliding 60-second window. Releases the lock while sleeping so
    other coroutines waiting on the same model can make progress.
    """
    window, lock = _ensure(model)
    limit = _LIMITS.get(model, _DEFAULT_RPM)

    while True:
        async with lock:
            now = time.monotonic()
            # Evict timestamps older than 60s
            while window and now - window[0] >= _WINDOW_S:
                window.popleft()

            if len(window) < limit:
                window.append(now)
                return

            # How long until the oldest call rolls off the window
            wait_s = _WINDOW_S - (now - window[0]) + 0.05

        logger.warning("Rate limit reached for %s (%d RPM) — sleeping %.1fs", model, limit, wait_s)
        await asyncio.sleep(wait_s)


def reset(model: str | None = None) -> None:
    """Clear rate-limit state. Used in tests."""
    if model is None:
        _windows.clear()
    elif model in _windows:
        _windows[model].clear()
