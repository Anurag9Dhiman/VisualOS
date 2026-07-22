"""Thin async wrapper around the MediaWiki REST API."""

from __future__ import annotations

import asyncio

import httpx

from src.contracts import ToolError, WikipediaResult

_TIMEOUT_S = 0.4
_BASE = "https://en.wikipedia.org/api/rest_v1"


async def wikipedia_search(query: str) -> WikipediaResult:
    search_url = "https://en.wikipedia.org/w/api.php"
    params: dict[str, str] = {"action": "query", "list": "search", "srsearch": query,
                              "srlimit": "1", "format": "json", "utf8": "1"}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            resp = await asyncio.wait_for(client.get(search_url, params=params), timeout=_TIMEOUT_S)
            resp.raise_for_status()
            hits = resp.json().get("query", {}).get("search", [])
            if not hits:
                raise ToolError("wikipedia_search", f"No results for '{query}'")
            title = hits[0]["title"]
            summary_resp = await asyncio.wait_for(
                client.get(f"{_BASE}/page/summary/{title.replace(' ', '_')}"),
                timeout=_TIMEOUT_S,
            )
            summary_resp.raise_for_status()
            sdata = summary_resp.json()
            return WikipediaResult(
                title=sdata.get("title", title),
                extract=sdata.get("extract", ""),
                url=sdata.get("content_urls", {}).get("desktop", {}).get("page", ""),
            )
    except TimeoutError as exc:
        raise ToolError("wikipedia_search", "timed out") from exc
    except httpx.HTTPError as exc:
        raise ToolError("wikipedia_search", str(exc)) from exc
