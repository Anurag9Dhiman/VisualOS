"""Memory Agent — Gemini embedding-based retrieval from SQLite."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

from google import genai

from src.contracts import CostEntry, MemoryHit, MemoryResult
from src.cost_logger import log_cost
from src import db, rate_limiter
from src.db import get_user_interests

logger = logging.getLogger("lens.memory")

_EMBED_MODEL = "text-embedding-004"
_TIMEOUT_S = 0.8


async def run_memory_agent(
    subject_name: str,
    user_id: str,
    cost_log: list[CostEntry],
) -> MemoryResult:
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    await rate_limiter.acquire(_EMBED_MODEL)
    embed_resp = await asyncio.wait_for(
        client.aio.models.embed_content(
            model=_EMBED_MODEL,
            contents=subject_name,
        ),
        timeout=_TIMEOUT_S / 2,
    )
    raw_embeddings = embed_resp.embeddings  # type: ignore[union-attr]
    query_vector: list[float] = raw_embeddings[0].values  # type: ignore[index, assignment]
    approx_tokens = max(1, len(subject_name.split()))
    cost_log.append(log_cost("memory_embed", _EMBED_MODEL, approx_tokens, 0))

    rows, interests = await asyncio.gather(
        asyncio.wait_for(db.search_interactions(user_id, query_vector, top_k=5), timeout=_TIMEOUT_S / 2),
        asyncio.wait_for(get_user_interests(user_id), timeout=_TIMEOUT_S / 2),
    )

    hits = [
        MemoryHit(
            interaction_id=r["interaction_id"],
            subject_name=r["subject_name"],
            summary=r["summary"],
            timestamp=r["timestamp"],
            similarity_score=r["similarity_score"],
        )
        for r in rows
        if r["similarity_score"] > 0.75
    ]
    logger.debug("Memory agent: %d hits, %d interests for user %s", len(hits), len(interests), user_id)
    return MemoryResult(hits=hits, user_id=user_id, user_interests_snapshot=interests)
