"""Tests for the FastAPI server — no API keys needed.

All pipeline calls are mocked. Tests cover:
  - GET /health
  - POST /analyze: happy path, bad MIME, missing image, timeout, no-card fallback
  - POST /analyze/stream: token events, final card event, timeout error event
  - Auth: correct key accepted, wrong key/no key → 401, health bypasses auth,
          dev-mode (LENS_API_KEY unset) allows all requests
"""

from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.contracts import NormalCard
from src.server import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 149, 237)).save(buf, format="JPEG")
    return buf.getvalue()


def _normal_card() -> NormalCard:
    return NormalCard(
        headline="Eiffel Tower",
        body="A 330-metre iron lattice tower in Paris, built 1887–1889.",
        personalized_hooks=[],
        citations=[],
        confidence_displayed="high",
        source_mix={"used_vision": True, "used_memory": False, "used_search": True},
        cost_usd_total=0.001,
        latency_ms=420,
    )


@pytest.fixture()
def client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


async def test_health_returns_ok(client: AsyncClient):
    async with client as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# POST /analyze — happy path
# ---------------------------------------------------------------------------


async def test_analyze_returns_normal_card(client: AsyncClient):
    mock_state = {"response_card": _normal_card(), "errors": []}
    with patch("src.server.run_pipeline", new_callable=AsyncMock, return_value=mock_state):
        async with client as c:
            r = await c.post(
                "/analyze",
                files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
                data={"lat": "12.95", "lng": "77.58", "user_id": "u1"},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["card_type"] == "normal"
    assert body["headline"] == "Eiffel Tower"


async def test_analyze_passes_lat_lng_to_pipeline(client: AsyncClient):
    mock_state = {"response_card": _normal_card(), "errors": []}
    with patch("src.server.run_pipeline", new_callable=AsyncMock, return_value=mock_state) as mock:
        async with client as c:
            await c.post(
                "/analyze",
                files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
                data={"lat": "48.858", "lng": "2.294"},
            )
    called_inp = mock.call_args[0][0]
    assert called_inp.lat == pytest.approx(48.858)
    assert called_inp.lng == pytest.approx(2.294)


async def test_analyze_defaults_user_id_to_anon(client: AsyncClient):
    mock_state = {"response_card": _normal_card(), "errors": []}
    with patch("src.server.run_pipeline", new_callable=AsyncMock, return_value=mock_state) as mock:
        async with client as c:
            await c.post(
                "/analyze",
                files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
    assert mock.call_args[0][0].user_id == "anon"


# ---------------------------------------------------------------------------
# POST /analyze — error cases
# ---------------------------------------------------------------------------


async def test_analyze_rejects_unsupported_mime(client: AsyncClient):
    async with client as c:
        r = await c.post(
            "/analyze",
            files={"image": ("clip.gif", b"GIF89a\x01\x00\x01\x00", "image/gif")},
        )
    assert r.status_code == 415
    assert "Unsupported" in r.json()["detail"]


async def test_analyze_missing_image_returns_422(client: AsyncClient):
    async with client as c:
        r = await c.post("/analyze", data={"lat": "12.95"})
    assert r.status_code == 422


async def test_analyze_timeout_returns_504(client: AsyncClient):
    with patch("src.server.run_pipeline", side_effect=asyncio.TimeoutError):
        async with client as c:
            r = await c.post(
                "/analyze",
                files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
    assert r.status_code == 504
    assert "timed out" in r.json()["detail"]


async def test_analyze_no_card_returns_500(client: AsyncClient):
    mock_state = {"response_card": None, "errors": ["vision failed"]}
    with patch("src.server.run_pipeline", new_callable=AsyncMock, return_value=mock_state):
        async with client as c:
            r = await c.post(
                "/analyze",
                files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
    assert r.status_code == 500


# ---------------------------------------------------------------------------
# POST /analyze/stream — SSE
# ---------------------------------------------------------------------------


async def test_stream_yields_token_events_then_card(client: AsyncClient):
    card = _normal_card()

    async def _mock_stream(inp):
        yield "The ", None
        yield "Eiffel ", None
        yield "Tower", {"response_card": card, "errors": []}

    with patch("src.server.stream_pipeline", side_effect=_mock_stream):
        async with client as c:
            r = await c.post(
                "/analyze/stream",
                files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )

    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]

    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    tokens = [e for e in events if e["type"] == "token"]
    cards = [e for e in events if e["type"] == "card"]

    # "Tower" is the full_text in the final (chunk, state) yield — it becomes
    # the card event, not a token event.
    assert [t["text"] for t in tokens] == ["The ", "Eiffel "]
    assert len(cards) == 1
    assert cards[0]["card"]["headline"] == "Eiffel Tower"
    assert cards[0]["card"]["card_type"] == "normal"


async def test_stream_rejects_unsupported_mime(client: AsyncClient):
    async with client as c:
        r = await c.post(
            "/analyze/stream",
            files={"image": ("doc.pdf", b"%PDF", "application/pdf")},
        )
    assert r.status_code == 415


async def test_stream_timeout_yields_error_event(client: AsyncClient):
    async def _mock_stream(inp):
        raise TimeoutError
        yield  # makes it an async generator

    with patch("src.server.stream_pipeline", side_effect=_mock_stream):
        async with client as c:
            r = await c.post(
                "/analyze/stream",
                files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )

    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert any(e["type"] == "error" for e in events)


# ---------------------------------------------------------------------------
# Auth — X-API-Key header
# ---------------------------------------------------------------------------

_VALID_KEY = "lens-test-secret-key"


async def test_auth_correct_key_accepted(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LENS_API_KEY", _VALID_KEY)
    mock_state = {"response_card": _normal_card(), "errors": []}
    with patch("src.server.run_pipeline", new_callable=AsyncMock, return_value=mock_state):
        async with client as c:
            r = await c.post(
                "/analyze",
                files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
                headers={"X-API-Key": _VALID_KEY},
            )
    assert r.status_code == 200


async def test_auth_wrong_key_returns_401(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LENS_API_KEY", _VALID_KEY)
    async with client as c:
        r = await c.post(
            "/analyze",
            files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
            headers={"X-API-Key": "wrong-key"},
        )
    assert r.status_code == 401
    assert "API key" in r.json()["detail"]


async def test_auth_missing_key_returns_401(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LENS_API_KEY", _VALID_KEY)
    async with client as c:
        r = await c.post(
            "/analyze",
            files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
    assert r.status_code == 401


async def test_auth_health_is_public(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LENS_API_KEY", _VALID_KEY)
    async with client as c:
        r = await c.get("/health")
    assert r.status_code == 200


async def test_auth_dev_mode_allows_requests(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LENS_API_KEY", raising=False)
    mock_state = {"response_card": _normal_card(), "errors": []}
    with patch("src.server.run_pipeline", new_callable=AsyncMock, return_value=mock_state):
        async with client as c:
            r = await c.post(
                "/analyze",
                files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
    assert r.status_code == 200


async def test_auth_stream_requires_key(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LENS_API_KEY", _VALID_KEY)
    async with client as c:
        r = await c.post(
            "/analyze/stream",
            files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
    assert r.status_code == 401
