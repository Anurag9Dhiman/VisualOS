"""Phase 0 CLI entry point.

Usage:
    python -m src.main --image path/to/photo.jpg --lat 48.8584 --lng 2.2945
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lens OS Phase 0 — visual identification pipeline")
    p.add_argument("--image", required=True, help="Path to the image file")
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lng", type=float, default=None)
    p.add_argument("--user-id", default="anon")
    p.add_argument("--json", action="store_true", help="Output raw JSON")
    return p.parse_args()


def _print_card(card_dict: dict) -> None:
    print()
    print("=" * 60)
    print(f"  {card_dict['headline']}")
    print("=" * 60)
    print()

    card_type = card_dict.get("card_type", "normal")

    if card_type == "fallback":
        if card_dict.get("observation"):
            print(card_dict["observation"])
            print()
        print(f"Suggestion: {card_dict.get('suggestion', '')}")
    else:
        print(card_dict.get("body", ""))
        print()
        hooks = card_dict.get("personalized_hooks", [])
        if hooks:
            print("Personalized:")
            for h in hooks[:3]:
                print(f"  • {h['fact']}")
            print()
        citations = card_dict.get("citations", [])
        if citations:
            print("Sources:", ", ".join(c["source_name"] for c in citations[:3]))

    cost = card_dict.get("cost_usd_total", 0)
    latency = card_dict.get("latency_ms", 0)
    conf = card_dict.get("confidence_displayed", card_dict.get("card_type", "?"))
    print(f"\nConfidence: {conf}  |  Cost: ${cost:.4f}  |  Latency: {latency}ms")
    print()


async def main() -> None:
    args = _parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found at {image_path}", file=sys.stderr)
        sys.exit(1)

    from src.contracts import LensInput
    from src.orchestrator import run_pipeline

    inp = LensInput(image_path=str(image_path), lat=args.lat, lng=args.lng, user_id=args.user_id)

    print(f"Processing {image_path.name}…", file=sys.stderr)
    try:
        state = await run_pipeline(inp)
    except asyncio.TimeoutError:
        print("Error: pipeline exceeded 2.5s overall timeout", file=sys.stderr)
        sys.exit(1)

    card = state.get("response_card")
    if card is None:
        print("Error: pipeline did not produce a response card", file=sys.stderr)
        for e in state.get("errors", []):
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    card_dict = card.model_dump()
    if args.json:
        print(json.dumps(card_dict, indent=2, default=str))
    else:
        _print_card(card_dict)


if __name__ == "__main__":
    asyncio.run(main())
