"""Thin async wrapper around the Wikidata SPARQL endpoint."""

from __future__ import annotations

import asyncio
import httpx
from src.contracts import ToolError, WikidataResult

_TIMEOUT_S = 0.4
_SPARQL_URL = "https://query.wikidata.org/sparql"
_HEADERS = {"Accept": "application/sparql-results+json", "User-Agent": "LensOS/0.1"}


async def wikidata_lookup(entity_id: str) -> WikidataResult:
    query = f"""
    SELECT ?propLabel ?valueLabel WHERE {{
      wd:{entity_id} ?prop ?value .
      ?prop wikibase:directClaim ?directProp .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }} LIMIT 20
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S, headers=_HEADERS) as client:
            resp = await asyncio.wait_for(
                client.get(_SPARQL_URL, params={"query": query, "format": "json"}),
                timeout=_TIMEOUT_S,
            )
            resp.raise_for_status()
            bindings = resp.json().get("results", {}).get("bindings", [])
            facts: dict[str, str] = {}
            for b in bindings:
                prop = b.get("propLabel", {}).get("value", "")
                value = b.get("valueLabel", {}).get("value", "")
                if prop and value and not prop.startswith("http"):
                    facts[prop] = value
            return WikidataResult(
                entity_id=entity_id,
                label=facts.pop("label", entity_id),
                facts=facts,
                url=f"https://www.wikidata.org/wiki/{entity_id}",
            )
    except asyncio.TimeoutError as exc:
        raise ToolError("wikidata_lookup", "timed out") from exc
    except httpx.HTTPError as exc:
        raise ToolError("wikidata_lookup", str(exc)) from exc
