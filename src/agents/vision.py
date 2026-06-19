"""Vision Agent — Gemini single-pass 5-step reasoning."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os

from google import genai
from google.genai import types

from src.contracts import CostEntry, VisionResult
from src.cost_logger import log_cost
from src.prompts import GeoPoint, VISION_SYSTEM_PROMPT

logger = logging.getLogger("lens.vision")

_MODEL = "gemini-2.0-flash"
_TIMEOUT_S = 0.8


async def run_vision_agent(
    image_b64: str,
    lat: float | None,
    lng: float | None,
    cost_log: list[CostEntry],
) -> VisionResult:
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    location = GeoPoint(lat=lat or 0.0, lng=lng or 0.0)
    prompt = (
        f"Identify the entity in this image.\n\n"
        f"Location: {location.lat}°N, {location.lng}°E\n"
        f"CoreML on-device hints: []\n\n"
        "Apply the 5-step reasoning process and return JSON."
    )

    img_bytes = base64.b64decode(image_b64)
    contents = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
    ]

    resp = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=VISION_SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=600,
            ),
        ),
        timeout=_TIMEOUT_S,
    )

    usage = resp.usage_metadata
    cost_log.append(log_cost("vision", _MODEL,
                             usage.prompt_token_count or 0,
                             usage.candidates_token_count or 0))

    data = json.loads(resp.text)
    return VisionResult(
        entity_name=data["entity_name"],
        entity_type=data.get("entity_type", "unknown"),
        confidence_level=data["confidence_level"],
        evidence=data.get("evidence", ["image analysis"]),
        alternatives=data.get("alternatives", []),
        failure_modes_checked=data.get("failure_modes_checked", ["lighting", "occlusion"]),
        needs_fallback=data.get("needs_fallback", False),
    )
