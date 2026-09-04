"""A2A (Agent-to-Agent) protocol routes for Lens OS.

Implements the Google A2A open spec so any A2A-compliant agent or client
can discover and call Lens OS over plain HTTP — no SDK required.

Spec: https://google.github.io/A2A/

Mounted into the main FastAPI app by src/server.py via include_router.

Endpoints:
    GET  /.well-known/agent.json   — agent discovery card
    POST /                          — JSON-RPC 2.0 task endpoint
        tasks/send                  → blocking, returns completed task
        tasks/sendSubscribe         → SSE stream of working→completed events
        tasks/get                   → not supported (stateless)
        tasks/cancel                → not supported (stateless)

Input format (tasks/send or tasks/sendSubscribe):
    message.parts must contain:
      - a FilePart  with the image  (type="file", file.mimeType, file.data=base64)
      - a TextPart  with JSON coords (type="text", text='{"lat":12.95,"lng":77.58}')

Output artifacts:
    name="response-card", parts=[{type="data", data={...ResponseCard...}}]
"""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.contracts import LensInput
from src.orchestrator import run_pipeline, stream_pipeline

logger = logging.getLogger("lens.a2a")

a2a_router = APIRouter(tags=["A2A"])

_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Agent card
# ---------------------------------------------------------------------------


def _build_agent_card() -> dict:
    return {
        "name": "Lens OS",
        "description": (
            "Visual intelligence agent — point a camera at any building, monument, "
            "statue, or object and get identification, history, and live info in under 3 seconds."
        ),
        "url": _BASE_URL,
        "version": "0.1.0",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "skills": [
            {
                "id": "visual-identify",
                "name": "Identify & Describe",
                "description": (
                    "Identifies an entity from a photo and returns a rich response card "
                    "with historical facts, live info, and personalised hooks."
                ),
                "tags": ["vision", "identification", "history", "travel"],
                "inputModes": ["image", "text"],
                "outputModes": ["text", "data"],
                "examples": [
                    "What is this building?",
                    "Tell me about this monument.",
                    '{"lat": 12.97, "lng": 77.59}',
                ],
            }
        ],
        "authentication": {"schemes": ["apiKey"]},
    }


@a2a_router.get("/.well-known/agent.json", include_in_schema=False)
async def agent_card() -> JSONResponse:
    return JSONResponse(_build_agent_card())


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


def _ok(rpc_id: str | None, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _err(rpc_id: str | None, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def _parse_message(message: dict) -> tuple[bytes, str, float | None, float | None, str]:
    """Return (image_bytes, mime_type, lat, lng, user_id) from an A2A message."""
    image_bytes = b""
    mime_type = "image/jpeg"
    lat: float | None = None
    lng: float | None = None
    user_id = "anon"

    for part in message.get("parts", []):
        ptype = part.get("type")
        if ptype == "file":
            f = part.get("file", {})
            mime_type = f.get("mimeType", "image/jpeg")
            image_bytes = base64.b64decode(f.get("data", ""))
        elif ptype == "text":
            try:
                meta = json.loads(part.get("text", "{}"))
                lat = meta.get("lat")
                lng = meta.get("lng")
                user_id = meta.get("user_id", "anon")
            except (json.JSONDecodeError, AttributeError):
                pass

    return image_bytes, mime_type, lat, lng, user_id


# ---------------------------------------------------------------------------
# tasks/send — blocking
# ---------------------------------------------------------------------------


async def _handle_send(task_id: str, message: dict) -> dict:
    image_bytes, mime_type, lat, lng, user_id = _parse_message(message)
    if not image_bytes:
        raise ValueError("No image found in message parts (expected a FilePart with base64 data)")

    suffix = ".png" if mime_type == "image/png" else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        state = await run_pipeline(
            LensInput(image_path=tmp_path, lat=lat, lng=lng, user_id=user_id)
        )
        card = state.get("response_card")
        if card is None:
            raise RuntimeError("Pipeline produced no card")
        return {
            "id": task_id,
            "status": {"state": "completed"},
            "artifacts": [
                {
                    "name": "response-card",
                    "parts": [{"type": "data", "data": card.model_dump()}],
                }
            ],
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# tasks/sendSubscribe — SSE streaming
# ---------------------------------------------------------------------------


async def _sse_stream(task_id: str, message: dict, rpc_id: str | None):
    image_bytes, mime_type, lat, lng, user_id = _parse_message(message)

    if not image_bytes:
        yield f"data: {json.dumps(_ok(rpc_id, {'id': task_id, 'status': {'state': 'failed'}, 'error': {'message': 'No image in message parts'}}))}\n\n"
        return

    suffix = ".png" if mime_type == "image/png" else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        yield f"data: {json.dumps(_ok(rpc_id, {'id': task_id, 'status': {'state': 'working'}}))}\n\n"

        async for chunk, state in stream_pipeline(
            LensInput(image_path=tmp_path, lat=lat, lng=lng, user_id=user_id)
        ):
            if state is None:
                event = _ok(
                    rpc_id,
                    {
                        "id": task_id,
                        "status": {"state": "working"},
                        "artifact": {"parts": [{"type": "text", "text": chunk}]},
                    },
                )
            else:
                card = state.get("response_card")
                event = _ok(
                    rpc_id,
                    {
                        "id": task_id,
                        "status": {"state": "completed"},
                        "artifact": {
                            "name": "response-card",
                            "parts": [
                                {"type": "data", "data": card.model_dump() if card else None}
                            ],
                            "lastChunk": True,
                        },
                    },
                )
            yield f"data: {json.dumps(event)}\n\n"

    except Exception as exc:
        logger.exception("tasks/sendSubscribe failed for task %s", task_id)
        yield f"data: {json.dumps(_ok(rpc_id, {'id': task_id, 'status': {'state': 'failed'}, 'error': {'message': str(exc)}}))}\n\n"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main task endpoint
# ---------------------------------------------------------------------------


@a2a_router.post("/", include_in_schema=False, response_model=None)
async def a2a_task(request: Request) -> JSONResponse | StreamingResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_err(None, -32700, "Parse error"), status_code=400)

    if body.get("jsonrpc") != "2.0":
        return JSONResponse(_err(None, -32600, "Invalid JSON-RPC 2.0 request"), status_code=400)

    rpc_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params", {})
    task_id = params.get("id") or str(uuid.uuid4())
    message = params.get("message", {})

    if method == "tasks/send":
        try:
            result = await _handle_send(task_id, message)
            return JSONResponse(_ok(rpc_id, result))
        except ValueError as exc:
            return JSONResponse(_err(rpc_id, -32602, str(exc)), status_code=400)
        except Exception as exc:
            logger.exception("tasks/send error")
            return JSONResponse(_err(rpc_id, -32603, str(exc)), status_code=500)

    if method == "tasks/sendSubscribe":
        return StreamingResponse(
            _sse_stream(task_id, message, rpc_id),
            media_type="text/event-stream",
        )

    if method in ("tasks/get", "tasks/cancel"):
        return JSONResponse(
            _err(rpc_id, -32601, f"{method} not supported — Lens OS is stateless"),
            status_code=501,
        )

    return JSONResponse(_err(rpc_id, -32601, f"Method not found: {method!r}"), status_code=404)
