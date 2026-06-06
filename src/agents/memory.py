"""Memory Agent — embedding-based retrieval from SQLite."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

from openai import AsyncOpenAI

from src.contracts import CostEntry, MemoryHit, MemoryResult
from src.cost_logger import log_cost
from src import db

logger = logging.getLogger("lens.memory")

_EMBED_MODEL = "text-embedding-3-small"
_TIMEOUT_S = 0.8


async def run_memory_agent(
    subject_name: str,
    user_id: str,
    cost_log: list[CostEntry],
) -> MemoryResult:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    embed_resp = await asyncio.wait_for(
        client.embeddings.create(model=_EMBED_MODEL, input=subject_name),
        timeout=_TIMEOUT_S / 2,
    )
    usage = embed_resp.usage
    cost_log.append(log_cost("memory_embed", _EMBED_MODEL, usage.prompt_tokens, 0))
    query_vector = embed_resp.data[0].embedding

    rows = await asyncio.wait_for(
        db.search_interactions(user_id, query_vector, top_k=5),
        timeout=_TIMEOUT_S / 2,
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
    logger.debug("Memory agent: %d hits for user %s on '%s'", len(hits), user_id, subject_name)
    return MemoryResult(hits=hits, user_id=user_id, user_interests_snapshot={})
