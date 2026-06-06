"""Thin async wrapper around the Tavily Search API."""

from __future__ import annotations

import asyncio
import os
import httpx
from src.contracts import TavilyResult, ToolError

_TIMEOUT_S = 0.5
_BASE = "https://api.tavily.com/search"


async def tavily_search(query: str) -> TavilyResult:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise ToolError("tavily_search", "TAVILY_API_KEY not set")
    payload = {"api_key": api_key, "query": query, "search_depth": "basic",
               "max_results": 5, "include_answer": True}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            resp = await asyncio.wait_for(client.post(_BASE, json=payload), timeout=_TIMEOUT_S)
            resp.raise_for_status()
            data = resp.json()
            results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                        "content": r.get("content", "")} for r in data.get("results", [])]
            return TavilyResult(query=query, results=results)
    except asyncio.TimeoutError as exc:
        raise ToolError("tavily_search", "timed out") from exc
    except httpx.HTTPError as exc:
        raise ToolError("tavily_search", str(exc)) from exc
