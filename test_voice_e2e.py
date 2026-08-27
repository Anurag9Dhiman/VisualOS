"""End-to-end voice integration test — runs against a local server.

Usage:
  # 1. Start the server in another terminal:
  #    source .venv/bin/activate && uvicorn src.server:app --port 8765

  # 2. Run this script with any photo:
  #    python test_voice_e2e.py path/to/photo.jpg

  # Optional flags:
  #    --lat 12.97 --lng 77.59   GPS coordinates (improves Vision accuracy)
  #    --url http://localhost:8765  if server is on a different port
"""

import argparse
import asyncio
import json
import sys
import time

import httpx
import websockets


async def run(image_path: str, lat: float | None, lng: float | None, base_url: str) -> None:
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")

    # ── Step 1: scan the image ────────────────────────────────────────────────
    print(f"\n[1/3] Scanning {image_path} ...")
    t0 = time.monotonic()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"

    async with httpx.AsyncClient(timeout=60.0) as client:
        data = {"user_id": "laptop-test", "user_locale": "en-IN"}
        if lat is not None:
            data["lat"] = str(lat)
        if lng is not None:
            data["lng"] = str(lng)

        resp = await client.post(
            f"{base_url}/analyze",
            data=data,
            files={"image": ("photo", image_bytes, mime)},
        )

    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text}")
        sys.exit(1)

    result = resp.json()
    session_id = result.get("session_id")
    card = result.get("card", {})
    latency = time.monotonic() - t0

    print(f"  Done in {latency:.1f}s")
    print(f"  Entity   : {card.get('headline', '(no headline)')}")
    print(f"  Card type: {card.get('card_type', '?')}")
    print(f"  Session  : {session_id}")

    if session_id is None:
        print(
            "  NOTE: session_id is None — this is a known limitation when the pipeline "
            "returns a cached card (Vision is skipped so no session is created).\n"
            "  Re-running the test with --no-cache flag clears the cache first.\n"
            "  Alternatively, pass --no-cache on the command line."
        )
        if not args.no_cache:
            print("  Hint: run with --no-cache to skip the cache and create a fresh session.")
        sys.exit(1)

    # ── Step 2: text voice Q&A ────────────────────────────────────────────────
    print("\n[2/3] Voice Q&A (text-only follow-up)...")

    async with websockets.connect(f"{ws_url}/v1/ws") as ws:
        await ws.send(json.dumps({
            "type": "session_start",
            "session_id": "ws-test-1",
            "user_id": "laptop-test",
        }))

        await ws.send(json.dumps({
            "type": "user_utterance",
            "text": "When was this built and who built it?",
            "router_class": "new_intent",
            "entity_refs": {"scan_session_id": session_id},
            "session_id": "ws-test-1",
        }))

        events = []
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=25.0)
                event = json.loads(raw)
                events.append(event)
                etype = event.get("type")
                print(f"  ← {etype}", end="")
                if etype == "speak":
                    print(f": {event.get('text', '')}")
                elif etype == "error":
                    print(f": {event.get('speak', event)}")
                else:
                    print()
                if etype in ("done", "error"):
                    break
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break

        await ws.send(json.dumps({"type": "session_end"}))

    # ── Step 3: region zoom Q&A ───────────────────────────────────────────────
    print("\n[3/3] Region zoom Q&A (top-left quarter of image)...")

    async with websockets.connect(f"{ws_url}/v1/ws") as ws:
        await ws.send(json.dumps({
            "type": "session_start",
            "session_id": "ws-test-2",
            "user_id": "laptop-test",
        }))

        await ws.send(json.dumps({
            "type": "user_utterance",
            "text": "What can you see in this part of the image?",
            "router_class": "new_intent",
            "entity_refs": {
                "scan_session_id": session_id,
                "region": {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.5},
            },
            "session_id": "ws-test-2",
        }))

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=25.0)
                event = json.loads(raw)
                etype = event.get("type")
                print(f"  ← {etype}", end="")
                if etype == "speak":
                    print(f": {event.get('text', '')}")
                elif etype == "error":
                    print(f": {event.get('speak', event)}")
                else:
                    print()
                if etype in ("done", "error"):
                    break
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break

        await ws.send(json.dumps({"type": "session_end"}))

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a JPEG or PNG photo")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lng", type=float, default=None)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Clear the response cache before scanning (ensures fresh Vision run + session_id)",
    )
    args = parser.parse_args()

    if args.no_cache:
        import sqlite3
        try:
            conn = sqlite3.connect("lens_memory.db")
            conn.execute("DELETE FROM response_cache")
            conn.commit()
            conn.close()
            print("Cache cleared.")
        except Exception as e:
            print(f"Warning: could not clear cache: {e}")

    asyncio.run(run(args.image, args.lat, args.lng, args.url))
