"""Memory Agent — 3-layer retrieval.

Layer 1  in-context rolling window of recent scans (src/memory_l1.py)
Layer 2  hybrid BM25 + semantic vector search (src/db.py)
Layer 3  persisted entity facts from past Search results (src/db.py)

All three layers run in parallel via asyncio.gather.
"""

from __future__ import annotations

import asyncio
import logging
import os

from google import genai
from langsmith import traceable

from src import db, memory_l1, rate_limiter
from src.contracts import CostEntry, MemoryHit, MemoryResult
from src.cost_logger import log_cost
from src.db import get_entity_facts, get_user_interests

logger = logging.getLogger("lens.memory")

_EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
_TIMEOUT_S = 5.0
_SIMILARITY_THRESHOLD = 0.75


@traceable(name="memory_agent")
async def run_memory_agent(
    subject_name: str,
    user_id: str,
    cost_log: list[CostEntry],
) -> MemoryResult:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Layer 1: get recent context immediately — no I/O needed
    recent_context = memory_l1.get_recent_context(user_id)

    # Embed the subject name (needed for Layer 2 semantic search)
    await rate_limiter.acquire(_EMBED_MODEL)
    embed_resp = await asyncio.wait_for(
        client.aio.models.embed_content(
            model=_EMBED_MODEL,
            contents=subject_name,
        ),
        timeout=_TIMEOUT_S / 2,
    )
    raw_embeddings = embed_resp.embeddings
    query_vector: list[float] = raw_embeddings[0].values  # type: ignore[index, assignment]
    approx_tokens = max(1, len(subject_name.split()))
    cost_log.append(log_cost("memory_embed", _EMBED_MODEL, approx_tokens, 0))

    # Layers 2 + 3 + interests in parallel
    rows, interests, facts = await asyncio.gather(
        asyncio.wait_for(
            db.search_interactions(user_id, query_vector, query_text=subject_name, top_k=5),
            timeout=_TIMEOUT_S / 2,
        ),
        asyncio.wait_for(get_user_interests(user_id), timeout=_TIMEOUT_S / 2),
        asyncio.wait_for(get_entity_facts(subject_name, top_k=10), timeout=_TIMEOUT_S / 2),
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
        if r["similarity_score"] > _SIMILARITY_THRESHOLD
    ]
    logger.debug(
        "Memory agent: %d hits, %d interests, %d entity_facts, recent_context=%r for user %s",
        len(hits),
        len(interests),
        len(facts),
        recent_context[:60] if recent_context else "",
        user_id,
    )
    return MemoryResult(
        hits=hits,
        user_id=user_id,
        user_interests_snapshot=interests,
        recent_context=recent_context,
        entity_facts=facts,
    )
