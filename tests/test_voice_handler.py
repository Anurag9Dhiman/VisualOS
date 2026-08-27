"""Tests for voice_handler and ws_server.

All Gemini calls are mocked. No API keys needed.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from PIL import Image

from src.contracts import (
    HistoricalFact,
    ImageRegion,
    LiveFact,
    ScanContext,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_scan_ctx(*, with_image: bool = False) -> ScanContext:
    """Build a minimal ScanContext for tests."""
    image_b64 = ""
    if with_image:
        buf = io.BytesIO()
        Image.new("RGB", (400, 300), color=(100, 149, 237)).save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

    now = datetime.now(UTC)
    return ScanContext(
        session_id=str(uuid4()),
        entity_name="India Gate",
        entity_type="monument",
        confidence_level="certain",
        card_headline="India Gate is a war memorial in New Delhi.",
        card_body="Built in 1931 to commemorate soldiers of the British Indian Army.",
        historical_facts=[
            HistoricalFact(fact="Designed by Edwin Lutyens", source="Wikipedia"),
            HistoricalFact(fact="Completed in 1931", source="Wikipedia"),
        ],
        live_facts=[LiveFact(fact="Open 24 hours", source="Travily", as_of="2024-01")],
        nearby_context="Located at Kartavya Path, New Delhi.",
        user_id="test-user",
        scanned_at=now,
        expires_at=now + timedelta(hours=1),
        image_b64=image_b64,
    )


def _mock_genai_response(text: str):
    resp = MagicMock()
    resp.text = text
    resp.usage_metadata = MagicMock(prompt_token_count=80, candidates_token_count=40)
    return resp


# ---------------------------------------------------------------------------
# voice_handler tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_answer_voice_query_text_only(monkeypatch):
    """Text-only Q&A returns the model's response and logs cost."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_resp = _mock_genai_response("India Gate was designed by Edwin Lutyens.")

    with (
        patch("src.voice_handler.genai.Client") as MockClient,
        patch("src.voice_handler.rate_limiter.acquire", new=AsyncMock()),
    ):
        MockClient.return_value.aio.models.generate_content = AsyncMock(return_value=mock_resp)
        from src.voice_handler import answer_voice_query

        cost_log = []
        ctx = _make_scan_ctx()
        answer = await answer_voice_query("Who designed India Gate?", ctx, None, cost_log)

    assert "Lutyens" in answer
    assert len(cost_log) == 1
    assert cost_log[0].agent == "voice_qa"


@pytest.mark.asyncio()
async def test_answer_voice_query_with_region(monkeypatch):
    """Region zoom path passes a cropped image to the model."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_resp = _mock_genai_response("The inscription reads INDIA GATE.")

    captured_contents = []

    async def _capture(*_a, contents=None, **_kw):
        captured_contents.append(contents)
        return mock_resp

    with (
        patch("src.voice_handler.genai.Client") as MockClient,
        patch("src.voice_handler.rate_limiter.acquire", new=AsyncMock()),
    ):
        MockClient.return_value.aio.models.generate_content = AsyncMock(side_effect=_capture)
        from src.voice_handler import answer_voice_query

        ctx = _make_scan_ctx(with_image=True)
        region = ImageRegion(x1=0.2, y1=0.1, x2=0.8, y2=0.5)
        answer = await answer_voice_query("What does the inscription say?", ctx, region, [])

    assert "INDIA GATE" in answer
    # Model was called with a list containing a Content with image part
    assert captured_contents, "model was never called"
    contents = captured_contents[0]
    assert isinstance(contents, list), "region path should pass list[Content]"


@pytest.mark.asyncio()
async def test_answer_voice_query_no_image_falls_back_to_text(monkeypatch):
    """When region is given but image_b64 is empty, falls back to text-only Q&A."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_resp = _mock_genai_response("I can see the arch area in the image.")

    captured_contents = []

    async def _capture(*_a, contents=None, **_kw):
        captured_contents.append(contents)
        return mock_resp

    with (
        patch("src.voice_handler.genai.Client") as MockClient,
        patch("src.voice_handler.rate_limiter.acquire", new=AsyncMock()),
    ):
        MockClient.return_value.aio.models.generate_content = AsyncMock(side_effect=_capture)
        from src.voice_handler import answer_voice_query

        ctx = _make_scan_ctx(with_image=False)  # no image stored
        region = ImageRegion(x1=0.2, y1=0.1, x2=0.8, y2=0.5)
        answer = await answer_voice_query("What's there?", ctx, region, [])

    assert answer  # got a response
    contents = captured_contents[0]
    # Should be a plain string (text-only path) since no image_b64 available
    assert isinstance(contents, str), "should fall back to text-only when image absent"


@pytest.mark.asyncio()
async def test_answer_voice_query_timeout(monkeypatch):
    """Timeout propagates so ws_server can catch and send an error event."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    async def _slow(*_a, **_kw):
        await asyncio.sleep(100)

    with (
        patch("src.voice_handler.genai.Client") as MockClient,
        patch("src.voice_handler.rate_limiter.acquire", new=AsyncMock()),
    ):
        MockClient.return_value.aio.models.generate_content = AsyncMock(side_effect=_slow)
        from src.voice_handler import answer_voice_query

        ctx = _make_scan_ctx()
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await asyncio.wait_for(answer_voice_query("test?", ctx, None, []), timeout=0.1)


# ---------------------------------------------------------------------------
# _crop_image unit test
# ---------------------------------------------------------------------------


def test_crop_image_returns_jpeg_bytes():
    """_crop_image produces JPEG bytes for a valid region."""
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), color=(200, 100, 50)).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    from src.voice_handler import _crop_image

    region = ImageRegion(x1=0.1, y1=0.1, x2=0.9, y2=0.9)
    crop_bytes = _crop_image(b64, region)

    assert len(crop_bytes) > 0
    # Verify it's a valid JPEG
    img = Image.open(io.BytesIO(crop_bytes))
    assert img.format == "JPEG"


def test_crop_image_clamps_to_image_bounds():
    """Coordinates outside [0,1] are clamped so PIL doesn't crash."""
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(0, 0, 0)).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    from src.voice_handler import _crop_image

    # x2/y2 capped at 1.0 by the Pydantic model, but test the crop itself is stable
    region = ImageRegion(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    crop_bytes = _crop_image(b64, region)
    assert len(crop_bytes) > 0


# ---------------------------------------------------------------------------
# ws_server integration tests (using ASGI test client)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_ws_rejects_wrong_first_message():
    """WebSocket closes with 4400 if first message is not session_start."""
    # _handle_query is not involved — test the guard directly via handle_voice_ws
    # by simulating the message sequence on a fake WebSocket.
    sent_codes: list[int] = []
    receive_calls = 0

    class FakeWS:
        async def accept(self) -> None:
            pass

        async def receive_json(self) -> dict:
            nonlocal receive_calls
            receive_calls += 1
            return {"type": "user_utterance", "text": "hello"}

        async def close(self, code: int = 1000) -> None:
            sent_codes.append(code)

        async def send_text(self, text: str) -> None:
            pass

    from src.ws_server import handle_voice_ws

    await handle_voice_ws(FakeWS())  # type: ignore[arg-type]
    assert 4400 in sent_codes, "should close with 4400 on bad first message"


@pytest.mark.asyncio()
async def test_ws_session_not_found_returns_error(monkeypatch):
    """WS returns an error event when scan_session_id is not in session store."""
    from src.session_store import _clear_all

    _clear_all()

    sent_events: list[dict] = []

    from src.ws_server import _handle_query

    class FakeWS:
        async def send_text(self, text: str) -> None:
            sent_events.append(json.loads(text))

    raw = {
        "type": "user_utterance",
        "text": "what year was it built?",
        "router_class": "new_intent",
        "entity_refs": {"scan_session_id": "nonexistent-session-id"},
        "session_id": "ws-session-id",
    }
    await _handle_query(FakeWS(), raw, "ws-session-id", "user-1")  # type: ignore[arg-type]

    types_sent = [e["type"] for e in sent_events]
    assert "ack" in types_sent
    assert "error" in types_sent


@pytest.mark.asyncio()
async def test_ws_query_happy_path(monkeypatch):
    """WS returns ack → progress → speak → done for a valid session + query."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from src.session_store import _clear_all, _store

    _clear_all()
    ctx = _make_scan_ctx()
    _store[ctx.session_id] = ctx  # inject directly

    sent_events: list[dict] = []

    mock_resp = _mock_genai_response("India Gate was built in 1931.")

    from src.ws_server import _handle_query

    class FakeWS:
        async def send_text(self, text: str) -> None:
            sent_events.append(json.loads(text))

    raw = {
        "type": "user_utterance",
        "text": "when was India Gate built?",
        "router_class": "new_intent",
        "entity_refs": {"scan_session_id": ctx.session_id},
        "session_id": "ws-abc",
    }

    with (
        patch("src.voice_handler.genai.Client") as MockClient,
        patch("src.voice_handler.rate_limiter.acquire", new=AsyncMock()),
    ):
        MockClient.return_value.aio.models.generate_content = AsyncMock(return_value=mock_resp)
        await _handle_query(FakeWS(), raw, "ws-abc", "user-1")  # type: ignore[arg-type]

    types_sent = [e["type"] for e in sent_events]
    assert types_sent == ["ack", "progress", "speak", "done"]
    speak_event = next(e for e in sent_events if e["type"] == "speak")
    assert "1931" in speak_event["text"]
    assert sent_events[-1]["outcome"] == "completed"


@pytest.mark.asyncio()
async def test_ws_query_with_region_sends_progress(monkeypatch):
    """WS sends a region-specific progress message when entity_refs contains a region."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from src.session_store import _clear_all, _store

    _clear_all()
    ctx = _make_scan_ctx(with_image=True)
    _store[ctx.session_id] = ctx

    sent_events: list[dict] = []
    mock_resp = _mock_genai_response("The inscription reads India Gate.")

    from src.ws_server import _handle_query

    class FakeWS:
        async def send_text(self, text: str) -> None:
            sent_events.append(json.loads(text))

    raw = {
        "type": "user_utterance",
        "text": "What is written there?",
        "entity_refs": {
            "scan_session_id": ctx.session_id,
            "region": {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.5},
        },
    }

    with (
        patch("src.voice_handler.genai.Client") as MockClient,
        patch("src.voice_handler.rate_limiter.acquire", new=AsyncMock()),
    ):
        MockClient.return_value.aio.models.generate_content = AsyncMock(return_value=mock_resp)
        await _handle_query(FakeWS(), raw, "ws-abc", "user-1")  # type: ignore[arg-type]

    types_sent = [e["type"] for e in sent_events]
    assert "ack" in types_sent
    assert "progress" in types_sent
    progress = next(e for e in sent_events if e["type"] == "progress")
    assert "Zooming" in progress["text"]


@pytest.mark.asyncio()
async def test_ws_query_with_invalid_region_is_ignored(monkeypatch):
    """Malformed region (non-numeric x1) is caught and silently ignored."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from src.session_store import _clear_all, _store

    _clear_all()
    ctx = _make_scan_ctx()
    _store[ctx.session_id] = ctx

    sent_events: list[dict] = []
    mock_resp = _mock_genai_response("Built in 1931.")

    from src.ws_server import _handle_query

    class FakeWS:
        async def send_text(self, text: str) -> None:
            sent_events.append(json.loads(text))

    raw = {
        "type": "user_utterance",
        "text": "When was it built?",
        "entity_refs": {
            "scan_session_id": ctx.session_id,
            # x1 violates ge=0.0 constraint → Pydantic ValidationError → except branch
            "region": {"x1": -99.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        },
    }

    with (
        patch("src.voice_handler.genai.Client") as MockClient,
        patch("src.voice_handler.rate_limiter.acquire", new=AsyncMock()),
    ):
        MockClient.return_value.aio.models.generate_content = AsyncMock(return_value=mock_resp)
        await _handle_query(FakeWS(), raw, "ws-abc", "user-1")  # type: ignore[arg-type]

    types_sent = [e["type"] for e in sent_events]
    assert "speak" in types_sent
    assert "done" in types_sent


@pytest.mark.asyncio()
async def test_ws_query_voice_handler_timeout_returns_error(monkeypatch):
    """WS sends a recoverable error event when voice_handler times out."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from src.session_store import _clear_all, _store

    _clear_all()
    ctx = _make_scan_ctx()
    _store[ctx.session_id] = ctx

    sent_events: list[dict] = []

    from src.ws_server import _handle_query

    class FakeWS:
        async def send_text(self, text: str) -> None:
            sent_events.append(json.loads(text))

    raw = {
        "type": "user_utterance",
        "text": "Tell me everything.",
        "entity_refs": {"scan_session_id": ctx.session_id},
    }

    with (
        patch("src.voice_handler.genai.Client") as MockClient,
        patch("src.voice_handler.rate_limiter.acquire", new=AsyncMock()),
        patch("src.ws_server.answer_voice_query", side_effect=TimeoutError),
    ):
        MockClient.return_value.aio.models.generate_content = AsyncMock(side_effect=TimeoutError)
        await _handle_query(FakeWS(), raw, "ws-abc", "user-1")  # type: ignore[arg-type]

    types_sent = [e["type"] for e in sent_events]
    assert "error" in types_sent
    err = next(e for e in sent_events if e["type"] == "error")
    assert err["recoverable"] is True


@pytest.mark.asyncio()
async def test_ws_query_voice_handler_exception_returns_error(monkeypatch):
    """WS sends a recoverable error event when voice_handler raises unexpectedly."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from src.session_store import _clear_all, _store

    _clear_all()
    ctx = _make_scan_ctx()
    _store[ctx.session_id] = ctx

    sent_events: list[dict] = []

    from src.ws_server import _handle_query

    class FakeWS:
        async def send_text(self, text: str) -> None:
            sent_events.append(json.loads(text))

    raw = {
        "type": "user_utterance",
        "text": "Who built this?",
        "entity_refs": {"scan_session_id": ctx.session_id},
    }

    with patch("src.ws_server.answer_voice_query", side_effect=RuntimeError("boom")):
        await _handle_query(FakeWS(), raw, "ws-abc", "user-1")  # type: ignore[arg-type]

    types_sent = [e["type"] for e in sent_events]
    assert "error" in types_sent
    err = next(e for e in sent_events if e["type"] == "error")
    assert err["recoverable"] is True


@pytest.mark.asyncio()
async def test_handle_voice_ws_full_loop(monkeypatch):
    """Full WS session: session_start → user_utterance → session_end closes cleanly."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from src.session_store import _clear_all, _store

    _clear_all()
    ctx = _make_scan_ctx()
    _store[ctx.session_id] = ctx

    messages = [
        {"type": "session_start", "session_id": "ws-loop-test", "user_id": "u1"},
        {
            "type": "user_utterance",
            "text": "who designed it?",
            "entity_refs": {"scan_session_id": ctx.session_id},
        },
        {"type": "session_end"},
    ]
    msg_iter = iter(messages)
    sent_events: list[dict] = []
    closed = []

    mock_resp = _mock_genai_response("Edwin Lutyens designed India Gate.")

    class FakeWS:
        async def accept(self) -> None:
            pass

        async def receive_json(self) -> dict:
            return next(msg_iter)

        async def send_text(self, text: str) -> None:
            sent_events.append(json.loads(text))

        async def close(self, code: int = 1000) -> None:
            closed.append(code)

    from src.ws_server import handle_voice_ws

    with (
        patch("src.voice_handler.genai.Client") as MockClient,
        patch("src.voice_handler.rate_limiter.acquire", new=AsyncMock()),
    ):
        MockClient.return_value.aio.models.generate_content = AsyncMock(return_value=mock_resp)
        await handle_voice_ws(FakeWS())  # type: ignore[arg-type]

    types_sent = [e["type"] for e in sent_events]
    assert "speak" in types_sent
    assert "done" in types_sent


@pytest.mark.asyncio()
async def test_handle_voice_ws_interrupt_event():
    """WS handles 'interrupt' event by sending ack and continuing."""
    messages = [
        {"type": "session_start", "session_id": "ws-int", "user_id": "u1"},
        {"type": "interrupt"},
        {"type": "session_end"},
    ]
    msg_iter = iter(messages)
    sent_events: list[dict] = []

    class FakeWS:
        async def accept(self) -> None:
            pass

        async def receive_json(self) -> dict:
            return next(msg_iter)

        async def send_text(self, text: str) -> None:
            sent_events.append(json.loads(text))

        async def close(self, code: int = 1000) -> None:
            pass

    from src.ws_server import handle_voice_ws

    await handle_voice_ws(FakeWS())  # type: ignore[arg-type]

    types_sent = [e["type"] for e in sent_events]
    assert "ack" in types_sent


@pytest.mark.asyncio()
async def test_handle_voice_ws_unknown_event_is_ignored():
    """Unknown event types are logged but do not crash the session."""
    messages = [
        {"type": "session_start", "session_id": "ws-unk", "user_id": "u1"},
        {"type": "totally_unknown_event"},
        {"type": "session_end"},
    ]
    msg_iter = iter(messages)

    class FakeWS:
        async def accept(self) -> None:
            pass

        async def receive_json(self) -> dict:
            return next(msg_iter)

        async def send_text(self, text: str) -> None:
            pass

        async def close(self, code: int = 1000) -> None:
            pass

    from src.ws_server import handle_voice_ws

    # Should complete without raising
    await handle_voice_ws(FakeWS())  # type: ignore[arg-type]


@pytest.mark.asyncio()
async def test_handle_voice_ws_receive_exception_exits_cleanly():
    """An exception from receive_json exits the loop without propagating."""
    call_count = 0

    class FakeWS:
        async def accept(self) -> None:
            pass

        async def receive_json(self) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"type": "session_start", "session_id": "ws-exc", "user_id": "u1"}
            raise RuntimeError("connection dropped")

        async def send_text(self, text: str) -> None:
            pass

        async def close(self, code: int = 1000) -> None:
            pass

    from src.ws_server import handle_voice_ws

    await handle_voice_ws(FakeWS())  # type: ignore[arg-type]
    # If we get here the exception was caught and the loop exited cleanly


@pytest.mark.asyncio()
async def test_send_swallows_websocket_disconnect():
    """_send silently swallows WebSocketDisconnect so the caller doesn't crash."""
    from fastapi import WebSocketDisconnect

    from src.ws_server import _send

    class FakeWS:
        async def send_text(self, text: str) -> None:
            raise WebSocketDisconnect(code=1001)

    # Should not raise
    await _send(FakeWS(), {"type": "speak", "text": "hello"})  # type: ignore[arg-type]


@pytest.mark.asyncio()
async def test_handle_voice_ws_first_message_timeout_exits_cleanly():
    """If the first message times out, handle_voice_ws returns without crashing."""
    from src.ws_server import handle_voice_ws

    class FakeWS:
        async def accept(self) -> None:
            pass

        async def receive_json(self) -> dict:
            raise TimeoutError("no first message")

        async def send_text(self, text: str) -> None:
            pass

        async def close(self, code: int = 1000) -> None:
            pass

    # Should return cleanly — no exception propagated
    await handle_voice_ws(FakeWS())  # type: ignore[arg-type]


@pytest.mark.asyncio()
async def test_handle_voice_ws_idle_timeout_exits_loop():
    """A TimeoutError from receive_json in the message loop closes the session."""
    call_count = 0

    class FakeWS:
        async def accept(self) -> None:
            pass

        async def receive_json(self) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"type": "session_start", "session_id": "ws-idle", "user_id": "u1"}
            raise TimeoutError("idle timeout")

        async def send_text(self, text: str) -> None:
            pass

        async def close(self, code: int = 1000) -> None:
            pass

    from src.ws_server import handle_voice_ws

    await handle_voice_ws(FakeWS())  # type: ignore[arg-type]
    assert call_count == 2  # first message + timeout on second


@pytest.mark.asyncio()
async def test_handle_voice_ws_close_exception_is_swallowed():
    """If websocket.close() raises, handle_voice_ws exits without propagating."""
    messages = [
        {"type": "session_start", "session_id": "ws-close-exc", "user_id": "u1"},
        {"type": "session_end"},
    ]
    msg_iter = iter(messages)

    class FakeWS:
        async def accept(self) -> None:
            pass

        async def receive_json(self) -> dict:
            return next(msg_iter)

        async def send_text(self, text: str) -> None:
            pass

        async def close(self, code: int = 1000) -> None:
            raise RuntimeError("connection already closed")

    from src.ws_server import handle_voice_ws

    # Should complete without raising despite close() throwing
    await handle_voice_ws(FakeWS())  # type: ignore[arg-type]


def test_crop_image_large_input_is_resized():
    """_crop_image thumbnails large crops to fit within 1024px (covers line 75)."""
    buf = io.BytesIO()
    # 2000×2000 image — crop covers the full image, so result is 2000×2000 → must thumbnail
    Image.new("RGB", (2000, 2000), color=(255, 0, 0)).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    from src.voice_handler import _crop_image

    region = ImageRegion(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    crop_bytes = _crop_image(b64, region)

    img = Image.open(io.BytesIO(crop_bytes))
    assert max(img.size) <= 1024, "large crop should be thumbnailed to ≤1024px"
