"""Vision Agent — GPT-4o single-pass 5-step reasoning."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path

from openai import AsyncOpenAI

from src.contracts import CostEntry, VisionResult
from src.cost_logger import log_cost
from src.prompts import GeoPoint, VISION_SYSTEM_PROMPT, build_vision_user_message

logger = logging.getLogger("lens.vision")

_MODEL = "gpt-4o"
_TIMEOUT_S = 0.8


async def run_vision_agent(
    image_b64: str,
    lat: float | None,
    lng: float | None,
    cost_log: list[CostEntry],
) -> VisionResult:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    frame_url = f"data:image/jpeg;base64,{image_b64}"
    location = GeoPoint(lat=lat or 0.0, lng=lng or 0.0)
    user_msg = build_vision_user_message(frame_url, location, [])

    resp = await asyncio.wait_for(
        client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            max_tokens=600,
        ),
        timeout=_TIMEOUT_S,
    )

    usage = resp.usage
    cost_log.append(log_cost("vision", _MODEL, usage.prompt_tokens, usage.completion_tokens))

    data = json.loads(resp.choices[0].message.content)
    return VisionResult(
        entity_name=data["entity_name"],
        entity_type=data.get("entity_type", "unknown"),
        confidence_level=data["confidence_level"],
        evidence=data.get("evidence", ["image analysis"]),
        alternatives=data.get("alternatives", []),
        failure_modes_checked=data.get("failure_modes_checked", ["lighting", "occlusion"]),
        needs_fallback=data.get("needs_fallback", False),
    )
