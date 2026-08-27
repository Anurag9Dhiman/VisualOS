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

from dotenv import load_dotenv

load_dotenv()  # must run before importing src modules that read os.environ at import time

from fastapi import (  # noqa: E402
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    Security,
    UploadFile,
    WebSocket,
)
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.security import APIKeyHeader  # noqa: E402
from slowapi import Limiter, _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.util import get_remote_address  # noqa: E402

from src.contracts import LensInput, NormalCard, ScanContext  # noqa: E402
from src.orchestrator import LensState, run_pipeline, stream_pipeline  # noqa: E402
from src.session_store import create_session, get_session, list_sessions  # noqa: E402
from src.ws_server import handle_voice_ws  # noqa: E402

logger = logging.getLogger("lens.server")

# ---------------------------------------------------------------------------
# Rate limiter — per-IP, applied to /analyze endpoints only.
# Default: 10 requests/minute. Override with LENS_RATE_LIMIT env var,
# e.g. LENS_RATE_LIMIT="30/minute" for a paid-tier deployment.
# ---------------------------------------------------------------------------
_RATE_LIMIT = os.getenv("LENS_RATE_LIMIT", "10/minute")
limiter = Limiter(key_func=get_remote_address, default_limits=[])


async def _on_shutdown() -> None:
    if os.environ.get("DATABASE_URL"):
        from src.db_postgres import close as pg_close

        await pg_close()
        logger.info("PostgreSQL pool closed")


app = FastAPI(title="Lens OS API", version="0.1.0", on_shutdown=[_on_shutdown])
app.state.limiter = limiter


async def _rate_limit_handler(request: Request, exc: Exception) -> Response:
    return _rate_limit_exceeded_handler(request, exc)  # type: ignore[arg-type]


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

_ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _store_session(state: LensState, user_id: str) -> ScanContext | None:
    """Build and store a ScanContext from a completed pipeline state."""
    vision = state.get("vision_result")
    card = state.get("response_card")
    search = state.get("search_result")
    if vision is None or card is None:
        return None
    card_body = card.body if isinstance(card, NormalCard) else card.observation
    historical_facts = search.historical_facts if search else []
    live_facts = search.live_facts if search else []
    nearby_context = search.nearby_context if search else ""
    return create_session(
        entity_name=vision.entity_name,
        entity_type=vision.entity_type,
        confidence_level=vision.confidence_level,
        card_headline=card.headline,
        card_body=card_body,
        historical_facts=historical_facts,
        live_facts=live_facts,
        nearby_context=nearby_context,
        user_id=user_id,
        image_b64=state.get("image_b64", ""),
    )


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
@limiter.limit(_RATE_LIMIT)
async def analyze(
    request: Request,
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

    session = _store_session(state, user_id)
    return {"card": card.model_dump(), "session_id": session.session_id if session else None}


@app.post("/analyze/stream", dependencies=[Security(_require_api_key)])
@limiter.limit(_RATE_LIMIT)
async def analyze_stream(
    request: Request,
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
                    session = _store_session(state, user_id)
                    yield f"data: {json.dumps({'type': 'card', 'card': card.model_dump() if card else None, 'session_id': session.session_id if session else None})}\n\n"
        except TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Pipeline timed out'})}\n\n"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return StreamingResponse(_events(), media_type="text/event-stream")


@app.websocket("/v1/ws")
async def voice_ws(websocket: WebSocket) -> None:
    """VoiceOS-compatible WebSocket endpoint.

    Speaks the voice_to_agent / agent_to_voice event contract (v1) so that
    VoiceOS (services/voice-service) can connect and forward voice queries
    directly to Lens OS as its CollectiveOS-compatible backend.

    Auth: intentionally unauthenticated — LiveKit's room token is the trust
    boundary on the VoiceOS side. Add X-API-Key enforcement when exposing
    beyond localhost.
    """
    await handle_voice_ws(websocket)


@app.get("/sessions", dependencies=[Security(_require_api_key)])
async def list_scan_sessions(
    user_id: str = "anon",
    limit: int = 20,
) -> list[dict]:
    """Return up to `limit` recent sessions for a user, newest first.

    Excludes image_b64 from each entry to keep response size small.
    """
    sessions = list_sessions(user_id, limit=min(limit, 50))
    return [s.model_dump(mode="json", exclude={"image_b64"}) for s in sessions]


@app.get("/session/{session_id}", dependencies=[Security(_require_api_key)])
async def get_scan_session(session_id: str) -> dict:
    """Return the full scan context for a completed pipeline run.

    Called by the voice AI repo to ground multi-turn follow-up conversations
    in the original scan output. Returns 404 if the session is unknown or
    has expired (TTL: 1 hour).
    """
    ctx = get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return ctx.model_dump(mode="json")
