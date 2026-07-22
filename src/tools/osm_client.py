"""Thin async wrapper around the Overpass API (OpenStreetMap)."""

from __future__ import annotations

import asyncio

import httpx

from src.contracts import OSMResult, ToolError

_TIMEOUT_S = 0.4
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


async def osm_lookup(lat: float, lng: float, radius_m: int = 50) -> OSMResult:
    query = f"""
    [out:json][timeout:3];
    (
      node(around:{radius_m},{lat},{lng})[name];
      way(around:{radius_m},{lat},{lng})[name];
      relation(around:{radius_m},{lat},{lng})[name];
    );
    out center 1;
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            resp = await asyncio.wait_for(
                client.post(_OVERPASS_URL, data={"data": query}), timeout=_TIMEOUT_S
            )
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            if not elements:
                return OSMResult(name=None, address=None, opening_hours=None, wheelchair=None)
            tags = elements[0].get("tags", {})
            address_parts = [tags.get("addr:housenumber", ""), tags.get("addr:street", ""),
                             tags.get("addr:city", ""), tags.get("addr:country", "")]
            address = ", ".join(p for p in address_parts if p) or None
            return OSMResult(name=tags.get("name"), address=address,
                             opening_hours=tags.get("opening_hours"),
                             wheelchair=tags.get("wheelchair"))
    except TimeoutError as exc:
        raise ToolError("osm_lookup", "timed out") from exc
    except httpx.HTTPError as exc:
        raise ToolError("osm_lookup", str(exc)) from exc
