"""Voice Q&A handler — answers voice follow-up questions about a scanned entity.

Two modes:
  - Text-only: user asks a follow-up question ("who built it?") with no region
  - Region zoom: user has pinched/tapped part of the image ("what's written
    on that arch?") — the mobile client passes fractional crop coordinates
    in entity_refs["region"]; we crop the stored image and re-run Vision
    on that region alongside the question.

In both modes the session context (facts, headline, body) is injected as
system context so the model answers in terms of the already-identified entity.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os

from google import genai
from google.genai import types
from PIL import Image

from src import rate_limiter
from src.contracts import CostEntry, ImageRegion, ScanContext
from src.cost_logger import log_cost

logger = logging.getLogger("lens.voice")

_MODEL = "gemini-3.6-flash"
_TIMEOUT_S = 12.0

_VOICE_SYSTEM_PROMPT = """You are a voice assistant for Lens OS. The user is standing in front of an entity that has already been identified and analysed. Your job is to answer follow-up questions concisely.

Rules:
- Answer in 1–3 short sentences. No more.
- Write as if speaking aloud — no markdown, no bullet points, no headers.
- If the answer is not in the context provided, say so honestly in one sentence rather than guessing.
- If a region crop is included, focus your answer on what is visible in that cropped area."""


def _context_block(ctx: ScanContext) -> str:
    """Build a compact text summary of the scan session for the prompt."""
    facts = "\n".join(f"- {f.fact}" for f in ctx.historical_facts[:4])
    live = "\n".join(f"- {f.fact}" for f in ctx.live_facts[:2])
    return (
        f"Entity: {ctx.entity_name} ({ctx.entity_type})\n"
        f"Headline: {ctx.card_headline}\n"
        f"Summary: {ctx.card_body}\n"
        f"Historical facts:\n{facts or '(none retrieved)'}\n"
        f"Live info:\n{live or '(none retrieved)'}\n"
        f"Nearby context: {ctx.nearby_context or '(not available)'}"
    )


def _crop_image(image_b64: str, region: ImageRegion) -> bytes:
    """Crop a base64-encoded image to the given fractional region.

    Returns JPEG bytes of the cropped area, resized to fit within 1024px
    so Vision handles it cleanly.
    """
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    box = (
        max(0, int(region.x1 * w)),
        max(0, int(region.y1 * h)),
        min(w, int(region.x2 * w)),
        min(h, int(region.y2 * h)),
    )
    cropped = img.crop(box)
    if max(cropped.size) > 1024:
        cropped.thumbnail((1024, 1024), Image.LANCZOS)  # type: ignore[attr-defined]
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


async def answer_voice_query(
    query: str,
    ctx: ScanContext,
    region: ImageRegion | None,
    cost_log: list[CostEntry],
) -> str:
    """Return a short, speakable answer to a voice query about the scanned entity.

    Args:
        query: The user's spoken question.
        ctx: The scan session context from session_store.
        region: Optional fractional crop region (from entity_refs["region"]).
                When present and image_b64 is available, the cropped image
                is sent to the model alongside the question.
        cost_log: Mutated in-place with the cost entry for this call.

    Returns:
        A 1–3 sentence answer suitable for TTS.
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    ctx_text = _context_block(ctx)

    if region and ctx.image_b64:
        crop_bytes = _crop_image(ctx.image_b64, region)
        contents: object = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"Scan context:\n{ctx_text}\n\n"
                            f"The user has zoomed into a specific region of the image "
                            f"(the cropped area is attached). "
                            f"Question: {query}"
                        )
                    ),
                    types.Part.from_bytes(data=crop_bytes, mime_type="image/jpeg"),
                ],
            )
        ]
    else:
        contents = f"Scan context:\n{ctx_text}\n\nQuestion: {query}"

    await rate_limiter.acquire(_MODEL)
    resp = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=_MODEL,
            contents=contents,  # type: ignore[arg-type]
            config=types.GenerateContentConfig(
                system_instruction=_VOICE_SYSTEM_PROMPT,
                max_output_tokens=200,
            ),
        ),
        timeout=_TIMEOUT_S,
    )

    usage = resp.usage_metadata
    cost_log.append(
        log_cost(
            "voice_qa",
            _MODEL,
            (usage.prompt_token_count or 0) if usage else 0,
            (usage.candidates_token_count or 0) if usage else 0,
        )
    )

    return (resp.text or "I'm not sure about that.").strip()
