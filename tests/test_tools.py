"""Tests for the four search tool clients — all HTTP is mocked.

Each client wraps httpx.AsyncClient with asyncio.wait_for. Tests verify:
  - Happy path: correct response is parsed into the right contract type
  - Empty / no-results response: safe defaults returned
  - asyncio.TimeoutError → ToolError("...", "timed out")
  - httpx.HTTPError     → ToolError("...", <message>)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.contracts import ToolError

# ---------------------------------------------------------------------------
# Helper — build a fake httpx response
# ---------------------------------------------------------------------------


def _httpx_resp(json_data: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _mock_client(get_resp=None, post_resp=None) -> MagicMock:
    """Return a mock that behaves as `async with httpx.AsyncClient() as client:`."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    if get_resp is not None:
        client.get = AsyncMock(return_value=get_resp)
    if post_resp is not None:
        client.post = AsyncMock(return_value=post_resp)
    return client


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

_WIKI_SEARCH_HIT = {"query": {"search": [{"title": "Eiffel Tower"}]}}
_WIKI_SUMMARY = {
    "title": "Eiffel Tower",
    "extract": "Iron lattice tower on the Champ de Mars in Paris.",
    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Eiffel_Tower"}},
}


@pytest.mark.asyncio
async def test_wikipedia_search_success():
    from src.tools.wikipedia_client import wikipedia_search

    mc = _mock_client()
    mc.get = AsyncMock(side_effect=[_httpx_resp(_WIKI_SEARCH_HIT), _httpx_resp(_WIKI_SUMMARY)])

    with patch("src.tools.wikipedia_client.httpx.AsyncClient", return_value=mc):
        result = await wikipedia_search("Eiffel Tower")

    assert result.title == "Eiffel Tower"
    assert "Paris" in result.extract
    assert "wikipedia.org" in result.url


@pytest.mark.asyncio
async def test_wikipedia_search_no_hits():
    from src.tools.wikipedia_client import wikipedia_search

    mc = _mock_client(get_resp=_httpx_resp({"query": {"search": []}}))

    with patch("src.tools.wikipedia_client.httpx.AsyncClient", return_value=mc):
        with pytest.raises(ToolError, match="No results"):
            await wikipedia_search("xyzzy-no-match")


@pytest.mark.asyncio
async def test_wikipedia_search_timeout():
    from src.tools.wikipedia_client import wikipedia_search

    mc = _mock_client()
    mc.get = AsyncMock(side_effect=TimeoutError())

    with patch("src.tools.wikipedia_client.httpx.AsyncClient", return_value=mc):
        with pytest.raises(ToolError, match="timed out"):
            await wikipedia_search("anything")


@pytest.mark.asyncio
async def test_wikipedia_search_http_error():
    from src.tools.wikipedia_client import wikipedia_search

    mc = _mock_client(get_resp=_httpx_resp({}, status=503))

    with patch("src.tools.wikipedia_client.httpx.AsyncClient", return_value=mc):
        with pytest.raises(ToolError):
            await wikipedia_search("anything")


# ---------------------------------------------------------------------------
# Wikidata
# ---------------------------------------------------------------------------

_WD_BINDINGS = {
    "results": {
        "bindings": [
            {"propLabel": {"value": "inception"}, "valueLabel": {"value": "1887"}},
            {"propLabel": {"value": "height"}, "valueLabel": {"value": "330 m"}},
            {"propLabel": {"value": "http://skip-me"}, "valueLabel": {"value": "ignored"}},
        ]
    }
}


@pytest.mark.asyncio
async def test_wikidata_lookup_success():
    from src.tools.wikidata_client import wikidata_lookup

    mc = _mock_client(get_resp=_httpx_resp(_WD_BINDINGS))

    with patch("src.tools.wikidata_client.httpx.AsyncClient", return_value=mc):
        result = await wikidata_lookup("Q243")

    assert result.entity_id == "Q243"
    assert result.facts["inception"] == "1887"
    assert result.facts["height"] == "330 m"
    assert "http://skip-me" not in result.facts  # URL props are filtered out
    assert "wikidata.org/wiki/Q243" in result.url


@pytest.mark.asyncio
async def test_wikidata_lookup_empty_bindings():
    from src.tools.wikidata_client import wikidata_lookup

    mc = _mock_client(get_resp=_httpx_resp({"results": {"bindings": []}}))

    with patch("src.tools.wikidata_client.httpx.AsyncClient", return_value=mc):
        result = await wikidata_lookup("Q999")

    assert result.entity_id == "Q999"
    assert result.facts == {}
    assert result.label == "Q999"  # falls back to entity_id when no label binding


@pytest.mark.asyncio
async def test_wikidata_lookup_timeout():
    from src.tools.wikidata_client import wikidata_lookup

    mc = _mock_client()
    mc.get = AsyncMock(side_effect=TimeoutError())

    with patch("src.tools.wikidata_client.httpx.AsyncClient", return_value=mc):
        with pytest.raises(ToolError, match="timed out"):
            await wikidata_lookup("Q1")


@pytest.mark.asyncio
async def test_wikidata_lookup_http_error():
    from src.tools.wikidata_client import wikidata_lookup

    mc = _mock_client(get_resp=_httpx_resp({}, status=500))

    with patch("src.tools.wikidata_client.httpx.AsyncClient", return_value=mc):
        with pytest.raises(ToolError):
            await wikidata_lookup("Q1")


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------

_TAVILY_RESPONSE = {
    "results": [
        {"title": "Eiffel Tower tickets", "url": "https://example.com/1", "content": "Book now."},
        {"title": "Eiffel Tower hours", "url": "https://example.com/2", "content": "Open daily."},
    ]
}


@pytest.mark.asyncio
async def test_tavily_search_success(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    from src.tools.tavily_client import tavily_search

    mc = _mock_client(post_resp=_httpx_resp(_TAVILY_RESPONSE))

    with patch("src.tools.tavily_client.httpx.AsyncClient", return_value=mc):
        result = await tavily_search("Eiffel Tower opening hours")

    assert result.query == "Eiffel Tower opening hours"
    assert len(result.results) == 2
    assert result.results[0]["title"] == "Eiffel Tower tickets"


@pytest.mark.asyncio
async def test_tavily_search_missing_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    import importlib

    from src.tools import tavily_client

    importlib.reload(tavily_client)  # reload so os.environ.get picks up change
    from src.tools.tavily_client import tavily_search

    with pytest.raises(ToolError, match="TAVILY_API_KEY not set"):
        await tavily_search("anything")


@pytest.mark.asyncio
async def test_tavily_search_timeout(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    from src.tools.tavily_client import tavily_search

    mc = _mock_client()
    mc.post = AsyncMock(side_effect=TimeoutError())

    with patch("src.tools.tavily_client.httpx.AsyncClient", return_value=mc):
        with pytest.raises(ToolError, match="timed out"):
            await tavily_search("anything")


@pytest.mark.asyncio
async def test_tavily_search_http_error(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    from src.tools.tavily_client import tavily_search

    mc = _mock_client(post_resp=_httpx_resp({}, status=429))

    with patch("src.tools.tavily_client.httpx.AsyncClient", return_value=mc):
        with pytest.raises(ToolError):
            await tavily_search("anything")


# ---------------------------------------------------------------------------
# OSM
# ---------------------------------------------------------------------------

_OSM_ELEMENT = {
    "elements": [
        {
            "tags": {
                "name": "Lalbagh West Gate",
                "addr:street": "Lalbagh Road",
                "addr:city": "Bangalore",
                "addr:country": "IN",
                "opening_hours": "Mo-Su 06:00-19:00",
                "wheelchair": "yes",
            }
        }
    ]
}


@pytest.mark.asyncio
async def test_osm_lookup_success():
    from src.tools.osm_client import osm_lookup

    mc = _mock_client(post_resp=_httpx_resp(_OSM_ELEMENT))

    with patch("src.tools.osm_client.httpx.AsyncClient", return_value=mc):
        result = await osm_lookup(12.9507, 77.5848, radius_m=50)

    assert result.name == "Lalbagh West Gate"
    assert "Lalbagh Road" in result.address
    assert result.opening_hours == "Mo-Su 06:00-19:00"
    assert result.wheelchair == "yes"


@pytest.mark.asyncio
async def test_osm_lookup_no_elements():
    from src.tools.osm_client import osm_lookup

    mc = _mock_client(post_resp=_httpx_resp({"elements": []}))

    with patch("src.tools.osm_client.httpx.AsyncClient", return_value=mc):
        result = await osm_lookup(0.0, 0.0)

    assert result.name is None
    assert result.address is None
    assert result.opening_hours is None


@pytest.mark.asyncio
async def test_osm_lookup_partial_address():
    from src.tools.osm_client import osm_lookup

    partial = {"elements": [{"tags": {"name": "Some Gate", "addr:city": "Delhi"}}]}
    mc = _mock_client(post_resp=_httpx_resp(partial))

    with patch("src.tools.osm_client.httpx.AsyncClient", return_value=mc):
        result = await osm_lookup(28.6139, 77.2090)

    assert result.name == "Some Gate"
    assert result.address == "Delhi"  # only city, no blank parts


@pytest.mark.asyncio
async def test_osm_lookup_timeout():
    from src.tools.osm_client import osm_lookup

    mc = _mock_client()
    mc.post = AsyncMock(side_effect=TimeoutError())

    with patch("src.tools.osm_client.httpx.AsyncClient", return_value=mc):
        with pytest.raises(ToolError, match="timed out"):
            await osm_lookup(0.0, 0.0)


@pytest.mark.asyncio
async def test_osm_lookup_http_error():
    from src.tools.osm_client import osm_lookup

    mc = _mock_client(post_resp=_httpx_resp({}, status=429))

    with patch("src.tools.osm_client.httpx.AsyncClient", return_value=mc):
        with pytest.raises(ToolError):
            await osm_lookup(0.0, 0.0)
