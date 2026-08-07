"""Vision Agent — Gemini single-pass 5-step reasoning with one-shot self-correction."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os

from google import genai
from google.genai import types
from PIL import Image
from pydantic import ValidationError

from src import rate_limiter
from src.contracts import CostEntry, VisionResult
from src.cost_logger import log_cost
from src.prompts import VISION_SYSTEM_PROMPT, GeoPoint

logger = logging.getLogger("lens.vision")

_MODEL = "gemini-2.0-flash"
_TIMEOUT_S = 0.8
_MAX_DIMENSION = 1024
_JPEG_QUALITY = 85


def _preprocess_image(image_b64: str) -> bytes:
    """Resize to max 1024px and compress to JPEG 85%. Reduces payload ~90% on large photos."""
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    original_size = len(raw)
    if max(img.size) > _MAX_DIMENSION:
        img.thumbnail((_MAX_DIMENSION, _MAX_DIMENSION), Image.LANCZOS)  # type: ignore[attr-defined]

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    processed = buf.getvalue()

    logger.debug(
        "Image preprocessed: %d KB → %d KB (%dx%d)",
        original_size // 1024,
        len(processed) // 1024,
        img.width,
        img.height,
    )
    return processed


def _parse_vision_response(text: str) -> VisionResult:
    """Parse and validate a raw JSON string into VisionResult. Raises on any failure."""
    data = json.loads(text)  # JSONDecodeError if malformed
    return VisionResult(
        entity_name=data["entity_name"],
        entity_type=data.get("entity_type", "unknown"),
        confidence_level=data["confidence_level"],
        evidence=data.get("evidence", ["image analysis"]),
        alternatives=data.get("alternatives", []),
        failure_modes_checked=data.get("failure_modes_checked", ["lighting", "occlusion"]),
        needs_fallback=data.get("needs_fallback", False),
    )


def _correction_message(bad_text: str, error: Exception) -> str:
    """Build a targeted correction prompt based on what went wrong."""
    if isinstance(error, json.JSONDecodeError):
        return (
            "Your last output was not valid JSON. "
            f"Parse error: {error}. "
            "Return ONLY the JSON object — no markdown, no prose, no code fences."
        )
    if isinstance(error, KeyError):
        return (
            f"Your last output was missing required field {error}. "
            "Return the complete JSON object with ALL required fields: "
            "entity_name, entity_type, confidence_level, evidence, "
            "alternatives, failure_modes_checked, needs_fallback."
        )
    # ValidationError (e.g. guessing without needs_fallback=True)
    return (
        f"Your last output failed schema validation: {error}. Fix the issue and return valid JSON."
    )


async def run_vision_agent(
    image_b64: str,
    lat: float | None,
    lng: float | None,
    cost_log: list[CostEntry],
) -> VisionResult:
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    location = GeoPoint(lat=lat or 0.0, lng=lng or 0.0)
    prompt = (
        "Identify the entity in this image.\n\n"
        f"Location: {location.lat}°N, {location.lng}°E\n"
        "CoreML on-device hints: []\n\n"
        "Apply the 5-step reasoning process and return JSON."
    )

    img_bytes = _preprocess_image(image_b64)

    # Build the base user turn once — reused across attempts.
    base_turn = types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        ],
    )

    bad_text = ""
    last_error: Exception | None = None

    for attempt in range(2):
        if attempt == 0:
            contents: list[types.Content] = [base_turn]
        else:
            # Feed the model's bad output back with a targeted correction.
            cost_log.append(log_cost("vision_retry", _MODEL, 0, 0))
            logger.info("Vision retry (attempt 2): %s", last_error)
            contents = [
                base_turn,
                types.Content(role="model", parts=[types.Part.from_text(text=bad_text)]),
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=_correction_message(bad_text, last_error))],  # type: ignore[arg-type]
                ),
            ]

        await rate_limiter.acquire(_MODEL)
        resp = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=_MODEL,
                contents=contents,  # type: ignore[arg-type]
                config=types.GenerateContentConfig(
                    system_instruction=VISION_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    max_output_tokens=600,
                ),
            ),
            timeout=_TIMEOUT_S,
        )

        usage = resp.usage_metadata
        cost_log.append(
            log_cost(
                "vision",
                _MODEL,
                (usage.prompt_token_count or 0) if usage else 0,
                (usage.candidates_token_count or 0) if usage else 0,
            )
        )

        bad_text = resp.text or ""

        try:
            return _parse_vision_response(bad_text)
        except (json.JSONDecodeError, KeyError, ValidationError) as exc:
            last_error = exc
            # Loop continues to attempt 2; after that, falls through to raise below.

    raise ValueError(f"Vision agent failed after 2 attempts: {last_error}") from last_error
