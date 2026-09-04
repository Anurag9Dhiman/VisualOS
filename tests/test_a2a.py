"""Tests for the A2A (Agent-to-Agent) protocol routes."""

from __future__ import annotations

import base64
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


def _jpeg_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 149, 237)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _normal_card() -> NormalCard:
    return NormalCard(
        headline="India Gate",
        body="A war memorial in New Delhi.",
        personalized_hooks=[],
        citations=[],
        confidence_displayed="high",
        source_mix={"used_vision": True, "used_memory": False, "used_search": True},
        cost_usd_total=0.001,
        latency_ms=800,
    )


def _send_body(task_id: str = "t1", extra_text: str | None = None) -> dict:
    parts: list[dict] = [{"type": "file", "file": {"mimeType": "image/jpeg", "data": _jpeg_b64()}}]
    if extra_text:
        parts.append({"type": "text", "text": extra_text})
    return {
        "jsonrpc": "2.0",
        "id": "rpc-1",
        "method": "tasks/send",
        "params": {"id": task_id, "message": {"parts": parts}},
    }


@pytest.fixture()
def client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# GET /.well-known/agent.json
# ---------------------------------------------------------------------------


async def test_agent_card_returns_200(client: AsyncClient):
    async with client as c:
        r = await c.get("/.well-known/agent.json")
    assert r.status_code == 200
    card = r.json()
    assert card["name"] == "Lens OS"
    assert card["capabilities"]["streaming"] is True
    assert len(card["skills"]) == 1
    assert card["skills"][0]["id"] == "visual-identify"


# ---------------------------------------------------------------------------
# tasks/send — happy path
# ---------------------------------------------------------------------------


async def test_tasks_send_returns_completed_task(client: AsyncClient):
    mock_state = {"response_card": _normal_card(), "errors": []}
    with patch("src.a2a.run_pipeline", new_callable=AsyncMock, return_value=mock_state):
        async with client as c:
            r = await c.post("/", json=_send_body())
    assert r.status_code == 200
    body = r.json()
    assert body["jsonrpc"] == "2.0"
    result = body["result"]
    assert result["status"]["state"] == "completed"
    assert result["artifacts"][0]["name"] == "response-card"
    data = result["artifacts"][0]["parts"][0]["data"]
    assert data["headline"] == "India Gate"


async def test_tasks_send_passes_gps_coords(client: AsyncClient):
    mock_state = {"response_card": _normal_card(), "errors": []}
    with patch("src.a2a.run_pipeline", new_callable=AsyncMock, return_value=mock_state) as mock:
        async with client as c:
            await c.post(
                "/",
                json=_send_body(extra_text='{"lat": 28.61, "lng": 77.23}'),
            )
    inp = mock.call_args[0][0]
    assert inp.lat == pytest.approx(28.61)
    assert inp.lng == pytest.approx(77.23)


# ---------------------------------------------------------------------------
# tasks/send — error cases
# ---------------------------------------------------------------------------


async def test_tasks_send_no_image_returns_400(client: AsyncClient):
    body = {
        "jsonrpc": "2.0",
        "id": "rpc-2",
        "method": "tasks/send",
        "params": {"id": "t2", "message": {"parts": [{"type": "text", "text": "hello"}]}},
    }
    async with client as c:
        r = await c.post("/", json=body)
    assert r.status_code == 400
    assert "error" in r.json()


async def test_tasks_send_invalid_jsonrpc_returns_400(client: AsyncClient):
    async with client as c:
        r = await c.post("/", json={"method": "tasks/send"})
    assert r.status_code == 400


async def test_tasks_send_unknown_method_returns_404(client: AsyncClient):
    body = {"jsonrpc": "2.0", "id": "rpc-x", "method": "tasks/fly", "params": {}}
    async with client as c:
        r = await c.post("/", json=body)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == -32601


async def test_tasks_get_returns_501(client: AsyncClient):
    body = {"jsonrpc": "2.0", "id": "rpc-3", "method": "tasks/get", "params": {"id": "t3"}}
    async with client as c:
        r = await c.post("/", json=body)
    assert r.status_code == 501


# ---------------------------------------------------------------------------
# tasks/sendSubscribe — SSE streaming
# ---------------------------------------------------------------------------


async def test_tasks_send_subscribe_streams_events(client: AsyncClient):
    card = _normal_card()

    async def _mock_stream(inp):
        yield "India ", None
        yield "Gate", {"response_card": card, "errors": []}

    with patch("src.a2a.stream_pipeline", side_effect=_mock_stream):
        async with client as c:
            body = {
                "jsonrpc": "2.0",
                "id": "rpc-4",
                "method": "tasks/sendSubscribe",
                "params": {
                    "id": "t4",
                    "message": {
                        "parts": [
                            {
                                "type": "file",
                                "file": {"mimeType": "image/jpeg", "data": _jpeg_b64()},
                            }
                        ]
                    },
                },
            }
            r = await c.post("/", json=body)

    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]

    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    states = [e["result"]["status"]["state"] for e in events]
    assert "working" in states
    assert states[-1] == "completed"
    last = events[-1]["result"]
    assert last["artifact"]["parts"][0]["data"]["headline"] == "India Gate"
