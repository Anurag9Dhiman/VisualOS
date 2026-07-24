"""Structured cost logger. Every LLM call must go through log_cost()."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from src.contracts import CostEntry

logger = logging.getLogger("lens.cost")

# Prices per 1 000 tokens (input / output)
_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.0-flash": (0.000075, 0.0003),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-1.5-pro": (0.00125, 0.005),
    "text-embedding-004": (0.000025, 0.0),
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = _PRICES.get(model, (0.001, 0.002))
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
        json.dumps(
            {
                "event": "llm_cost",
                "agent": agent,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost, 6),
            }
        )
    )
    return entry
