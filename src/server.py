"""
FastAPI server — wraps the Lens OS pipeline behind a REST API.

Endpoints:
  GET  /health          → {"status": "ok", "version": "0.1.0"}
  POST /analyze         → ResponseCard JSON (blocking, ≤2.5s)
  POST /analyze/stream  → SSE: token events then a final card event
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

from src.contracts import LensInput
from src.orchestrator import run_pipeline, stream_pipeline

logger = logging.getLogger("lens.server")

app = FastAPI(title="Lens OS API", version="0.1.0")

_ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _require_api_key(key: str | None = Security(_api_key_header)) -> None:
    """Validate X-API-Key header against LENS_API_KEY env var.

    If LENS_API_KEY is unset the server runs in dev mode — all requests are
    allowed, but a warning is logged on every call as a reminder to set the
    key before deploying.
    """
    configured = os.getenv("LENS_API_KEY")
    if configured is None:
        logger.warning("LENS_API_KEY not set — running without auth (dev mode only)")
        return
    if key is None or not hmac.compare_digest(key, configured):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/analyze", dependencies=[Security(_require_api_key)])
async def analyze(
    image: UploadFile = File(..., description="JPEG, PNG, or WebP photo"),
    lat: float | None = Form(None, description="GPS latitude"),
    lng: float | None = Form(None, description="GPS longitude"),
    user_id: str = Form("anon", description="Stable user identifier"),
    user_locale: str = Form("en-IN", description="BCP-47 locale tag"),
) -> dict:
    if image.content_type not in _ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported image type: {image.content_type}")

    image_bytes = await image.read()
    suffix = ".png" if image.content_type == "image/png" else ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    inp = LensInput(image_path=tmp_path, lat=lat, lng=lng, user_id=user_id, user_locale=user_locale)
    try:
        state = await run_pipeline(inp)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Pipeline timed out — try again") from None
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    card = state.get("response_card")
    if card is None:
        raise HTTPException(status_code=500, detail="Pipeline produced no card")

    return card.model_dump()


@app.post("/analyze/stream", dependencies=[Security(_require_api_key)])
async def analyze_stream(
    image: UploadFile = File(..., description="JPEG, PNG, or WebP photo"),
    lat: float | None = Form(None),
    lng: float | None = Form(None),
    user_id: str = Form("anon"),
    user_locale: str = Form("en-IN"),
) -> StreamingResponse:
    if image.content_type not in _ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported image type: {image.content_type}")

    image_bytes = await image.read()
    suffix = ".png" if image.content_type == "image/png" else ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    inp = LensInput(image_path=tmp_path, lat=lat, lng=lng, user_id=user_id, user_locale=user_locale)

    async def _events():
        try:
            async for chunk, state in stream_pipeline(inp):
                if state is None:
                    yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
                else:
                    card = state.get("response_card")
                    yield f"data: {json.dumps({'type': 'card', 'card': card.model_dump() if card else None})}\n\n"
        except TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Pipeline timed out'})}\n\n"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return StreamingResponse(_events(), media_type="text/event-stream")
