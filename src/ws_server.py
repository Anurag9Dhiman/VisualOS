"""WebSocket server implementing the VoiceOS agent contract (v1).

VoiceOS's voice-service connects here (ws://host/v1/ws) when its router
classifies an utterance as new_intent or session_query. Lens OS acts as
the CollectiveOS-compatible backend, answering voice queries about scanned
entities and supporting region-zoom follow-ups.

Event flow per query:
  client → session_start        (first message — must be this)
  client → user_utterance       (new_intent: follow-up Q about the scan)
  server → ack                  (immediately — keeps VoiceOS from timing out)
  server → progress             (optional — "looking at that area...")
  server → speak                (the actual answer, high-priority)
  server → done                 (signals task complete to VoiceOS)
  client → session_end          (clean disconnect)

Zoom / region queries:
  VoiceOS sends entity_refs: {"region": {"x1": 0.3, "y1": 0.1, "x2": 0.7, "y2": 0.4}}
  alongside the utterance when the user has pinched/tapped a specific area
  on the mobile screen. Lens OS crops the stored image to that region and
  passes it to the vision model for a focused answer.

Session resolution:
  The voice WS session_id (UUID from VoiceOS) is separate from the Lens OS
  scan session_id (UUID from /analyze). The mobile app is expected to pass
  the scan session_id in entity_refs["scan_session_id"] on the first
  user_utterance so we can look up the right ScanContext. As a fallback we
  also try the WS session_id directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from src.contracts import CostEntry, ImageRegion
from src.session_store import get_session
from src.voice_handler import answer_voice_query

logger = logging.getLogger("lens.ws")


async def _send(ws: WebSocket, event: dict) -> None:
    try:
        await ws.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass


async def handle_voice_ws(websocket: WebSocket) -> None:
    """Main handler for a VoiceOS WebSocket session."""
    await websocket.accept()

    # ── First message must be session_start ──────────────────────────────────
    try:
        first_raw = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
    except (TimeoutError, WebSocketDisconnect, Exception):
        return

    if first_raw.get("type") != "session_start":
        await websocket.close(code=4400)
        return

    ws_session_id: str = first_raw.get("session_id", "")
    user_id: str = first_raw.get("user_id", "anon")
    logger.info("Voice WS session opened: ws_sid=%s user=%s", ws_session_id, user_id)

    # ── Message loop ──────────────────────────────────────────────────────────
    while True:
        try:
            raw = await asyncio.wait_for(websocket.receive_json(), timeout=300.0)
        except TimeoutError:
            logger.info("Voice WS idle timeout — closing ws_sid=%s", ws_session_id)
            break
        except (WebSocketDisconnect, Exception):
            break

        event_type: str = raw.get("type", "")

        if event_type == "session_end":
            logger.info("Voice WS session ended cleanly: ws_sid=%s", ws_session_id)
            break

        if event_type in ("user_utterance", "session_query"):
            await _handle_query(websocket, raw, ws_session_id, user_id)
            continue

        if event_type == "interrupt":
            task_id = str(uuid4())
            await _send(websocket, {"type": "ack", "task_id": task_id, "text": "Got it."})
            continue

        # Unknown event type — log and ignore; additionalProperties: true in schema
        logger.debug("Ignoring unknown event type: %r", event_type)

    try:
        await websocket.close()
    except Exception:
        pass


async def _handle_query(
    ws: WebSocket,
    raw: dict,
    ws_session_id: str,
    user_id: str,
) -> None:
    """Process one user_utterance or session_query and stream back results."""
    task_id = str(uuid4())
    query_text: str = raw.get("text") or raw.get("query") or ""

    # ── Extract zoom region from entity_refs ──────────────────────────────────
    entity_refs: dict = raw.get("entity_refs") or {}
    region_data = entity_refs.get("region")
    region: ImageRegion | None = None
    if region_data:
        try:
            region = ImageRegion(**region_data)
        except Exception:
            logger.warning("Invalid region in entity_refs: %r — ignoring", region_data)

    # ── Ack immediately so VoiceOS doesn't time out ───────────────────────────
    ack_text = "Looking at that area..." if region else "On it..."
    await _send(ws, {"type": "ack", "task_id": task_id, "text": ack_text})

    # ── Resolve ScanContext ───────────────────────────────────────────────────
    # Mobile app should pass scan_session_id in entity_refs; fall back to ws_session_id.
    scan_session_id: str = entity_refs.get("scan_session_id", "") or ws_session_id
    ctx = get_session(scan_session_id)
    if ctx is None:
        await _send(
            ws,
            {
                "type": "error",
                "task_id": task_id,
                "recoverable": True,
                "speak": (
                    "I don't have context for this scan yet. "
                    "Please scan something first and make sure the session ID is passed."
                ),
            },
        )
        return

    # ── Progress hint (low-priority, VoiceOS can drop if user is speaking) ───
    if region:
        await _send(
            ws,
            {
                "type": "progress",
                "task_id": task_id,
                "text": f"Zooming into that region of {ctx.entity_name}...",
                "priority": "low",
            },
        )
    else:
        await _send(
            ws,
            {
                "type": "progress",
                "task_id": task_id,
                "text": f"Looking up information about {ctx.entity_name}...",
                "priority": "low",
            },
        )

    # ── Run voice Q&A ─────────────────────────────────────────────────────────
    cost_log: list[CostEntry] = []
    try:
        answer = await answer_voice_query(query_text, ctx, region, cost_log)
    except TimeoutError:
        logger.warning("voice_handler timed out for task %s", task_id)
        await _send(
            ws,
            {
                "type": "error",
                "task_id": task_id,
                "recoverable": True,
                "speak": "That took too long. Try asking again.",
            },
        )
        return
    except Exception as exc:
        logger.error("voice_handler error task=%s: %s", task_id, exc)
        await _send(
            ws,
            {
                "type": "error",
                "task_id": task_id,
                "recoverable": True,
                "speak": "Something went wrong. Try asking again.",
            },
        )
        return

    # ── Stream answer back ────────────────────────────────────────────────────
    await _send(
        ws,
        {
            "type": "speak",
            "task_id": task_id,
            "text": answer,
            "priority": "high",
        },
    )
    await _send(
        ws,
        {
            "type": "done",
            "task_id": task_id,
            "outcome": "completed",
            "summary_speak": answer,
        },
    )
    logger.info(
        "Voice query answered: task=%s entity=%s region=%s cost=$%.6f",
        task_id,
        ctx.entity_name,
        bool(region),
        sum(e.cost_usd for e in cost_log),
    )
