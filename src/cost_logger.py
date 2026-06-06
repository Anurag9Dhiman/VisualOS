"""Structured cost logger. Every LLM call must go through log_cost()."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from src.contracts import CostEntry

logger = logging.getLogger("lens.cost")

_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-2024-08-06": (0.0025, 0.010),
    "claude-sonnet-4-5": (0.003, 0.015),
    "claude-sonnet-4-6": (0.003, 0.015),
    "text-embedding-3-small": (0.00002, 0.0),
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = _PRICES.get(model, (0.01, 0.03))
    return (input_tokens / 1000) * prices[0] + (output_tokens / 1000) * prices[1]


def log_cost(agent: str, model: str, input_tokens: int, output_tokens: int) -> CostEntry:
    cost = compute_cost(model, input_tokens, output_tokens)
    entry = CostEntry(
        agent=agent,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        timestamp=datetime.utcnow(),
    )
    logger.info(
        json.dumps({
            "event": "llm_cost",
            "agent": agent,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })
    )
    return entry
