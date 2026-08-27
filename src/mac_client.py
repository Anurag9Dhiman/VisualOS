"""Lens OS — macOS client.

Captures images from four sources on a MacBook and streams them through
the Lens OS pipeline, printing the analysis to the terminal.

Usage:
  python -m src.mac_client screen     # select region on screen (like Cmd+Shift+4)
  python -m src.mac_client webcam     # snap from built-in camera
  python -m src.mac_client clip       # analyse image currently on clipboard
  python -m src.mac_client file PATH  # analyse any image file

The server must be running:
  uvicorn src.server:app --port 8765

Options:
  --lat  FLOAT  override GPS latitude  (default: IP geolocation)
  --lng  FLOAT  override GPS longitude
  --url  STR    server base URL        (default: http://127.0.0.1:8765)
  --json        print raw JSON card instead of streaming text
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------


def _ip_location() -> tuple[float | None, float | None]:
    """Best-effort city-level location from public IP. Returns (None, None) on failure."""
    try:
        resp = httpx.get("https://ipapi.co/json/", timeout=4.0)
        data = resp.json()
        return float(data["latitude"]), float(data["longitude"])
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Capture helpers — each returns raw JPEG bytes
# ---------------------------------------------------------------------------


def capture_screen() -> bytes:
    """Interactive screen-region capture (like Cmd+Shift+4).

    Uses the built-in macOS `screencapture` command — no extra dependencies.
    The user clicks and drags to select a region, then the JPEG is returned.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    print("Select a region on your screen (click and drag)…")
    result = subprocess.run(
        ["screencapture", "-i", "-s", "-t", "jpg", tmp_path],
        capture_output=True,
    )
    if result.returncode != 0 or not Path(tmp_path).exists():
        raise RuntimeError("Screen capture cancelled or failed.")

    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)

    if not data:
        raise RuntimeError("Screen capture produced an empty file — did you select a region?")
    return data


def capture_webcam(camera_index: int = 0) -> bytes:
    """Snap a single frame from the built-in camera."""
    import cv2  # lazy import — only needed for this path

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check System Preferences → Privacy → Camera.")

    print("Camera ready. Press SPACE to capture, Q to cancel.")
    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Lens OS — Webcam  (SPACE = capture, Q = quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            captured = frame
            break
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured is None:
        raise RuntimeError("Capture cancelled.")

    ok, buf = cv2.imencode(".jpg", captured, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Failed to encode webcam frame as JPEG.")
    return buf.tobytes()


def capture_clipboard() -> bytes:
    """Grab the image currently on the system clipboard."""
    try:
        from PIL import ImageGrab  # Pillow — already in requirements
    except ImportError:
        raise RuntimeError("Pillow is required for clipboard capture: pip install Pillow")

    img = ImageGrab.grabclipboard()
    if img is None:
        raise RuntimeError("No image found on clipboard. Copy an image first (Cmd+C).")

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def load_file(path: str) -> bytes:
    """Load any image file (JPEG, PNG, WebP) from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_bytes()


def _mime_for(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")


# ---------------------------------------------------------------------------
# Stream analysis
# ---------------------------------------------------------------------------


def stream_analysis(
    image_bytes: bytes,
    filename: str = "capture.jpg",
    mime: str = "image/jpeg",
    lat: float | None = None,
    lng: float | None = None,
    user_id: str = "mac-user",
    base_url: str = "http://127.0.0.1:8765",
    raw_json: bool = False,
) -> None:
    """Send image to /analyze/stream and print result."""

    if raw_json:
        with httpx.Client(timeout=120.0) as client:
            data: dict = {"user_id": user_id, "user_locale": "en-IN"}
            if lat is not None:
                data["lat"] = str(lat)
            if lng is not None:
                data["lng"] = str(lng)
            resp = client.post(
                f"{base_url}/analyze",
                data=data,
                files={"image": (filename, image_bytes, mime)},
            )
            if resp.status_code != 200:
                print(f"Error {resp.status_code}: {resp.text}", file=sys.stderr)
                sys.exit(1)
            print(json.dumps(resp.json(), indent=2))
        return

    # Streaming path
    data = {"user_id": user_id, "user_locale": "en-IN"}
    if lat is not None:
        data["lat"] = str(lat)
    if lng is not None:
        data["lng"] = str(lng)

    print("\n" + "─" * 60)
    t0 = time.monotonic()
    card_data = None

    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{base_url}/analyze/stream",
            data=data,
            files={"image": (filename, image_bytes, mime)},
        ) as resp:
            if resp.status_code != 200:
                body = resp.read()
                print(f"Error {resp.status_code}: {body.decode()}", file=sys.stderr)
                sys.exit(1)

            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                event_type = payload.get("type")

                if event_type == "token":
                    print(payload.get("text", ""), end="", flush=True)

                elif event_type == "card":
                    card_data = payload.get("card", {})
                    session_id = payload.get("session_id")
                    elapsed = time.monotonic() - t0
                    print(f"\n\n{'─' * 60}")
                    print(f"Latency : {elapsed:.1f}s")
                    print(f"Card    : {card_data.get('card_type', '?')}")
                    if session_id:
                        print(f"Session : {session_id}")
                        print(
                            f"\nVoice follow-up: connect VoiceOS to ws://{base_url.split('//')[1]}/v1/ws"
                            f"\nand pass scan_session_id={session_id!r} in entity_refs."
                        )

                elif event_type == "error":
                    print(f"\nError: {payload.get('detail', payload)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lens OS macOS client — point your Mac at anything.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Capture modes:
  screen   Select a region on your screen (like Cmd+Shift+4)
  webcam   Snap from the built-in FaceTime camera
  clip     Analyse image currently on clipboard (after Cmd+C)
  file     Analyse any image file on disk

Examples:
  python -m src.mac_client screen
  python -m src.mac_client webcam
  python -m src.mac_client clip
  python -m src.mac_client file ~/Downloads/monument.jpg --lat 28.61 --lng 77.20
""",
    )
    parser.add_argument(
        "mode",
        choices=["screen", "webcam", "clip", "file"],
        help="Input source",
    )
    parser.add_argument("path", nargs="?", help="Image path (required for 'file' mode)")
    parser.add_argument("--lat", type=float, default=None, help="GPS latitude override")
    parser.add_argument("--lng", type=float, default=None, help="GPS longitude override")
    parser.add_argument("--url", default="http://127.0.0.1:8765", help="Server base URL")
    parser.add_argument("--user", default="mac-user", help="User ID for memory")
    parser.add_argument("--json", dest="raw_json", action="store_true", help="Print raw JSON card")
    args = parser.parse_args()

    # -- Capture -------------------------------------------------------------
    image_bytes: bytes
    filename = "capture.jpg"
    mime = "image/jpeg"

    try:
        if args.mode == "screen":
            image_bytes = capture_screen()
        elif args.mode == "webcam":
            image_bytes = capture_webcam()
        elif args.mode == "clip":
            image_bytes = capture_clipboard()
        elif args.mode == "file":
            if not args.path:
                parser.error("'file' mode requires a PATH argument")
            image_bytes = load_file(args.path)
            filename = Path(args.path).name
            mime = _mime_for(args.path)
        else:
            parser.error(f"Unknown mode: {args.mode}")
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Captured {len(image_bytes) // 1024} KB via '{args.mode}' mode.")

    # -- Location ------------------------------------------------------------
    lat, lng = args.lat, args.lng
    if lat is None or lng is None:
        print("Resolving location from IP…", end=" ", flush=True)
        ip_lat, ip_lng = _ip_location()
        if ip_lat is not None:
            lat = lat or ip_lat
            lng = lng or ip_lng
            print(f"({lat:.3f}, {lng:.3f}) — city-level precision")
        else:
            print("unavailable — sending without GPS")

    # -- Check server --------------------------------------------------------
    try:
        httpx.get(f"{args.url}/health", timeout=3.0)
    except Exception:
        print(
            f"\nServer not reachable at {args.url}.\n"
            "Start it with: uvicorn src.server:app --port 8765",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Analyse -------------------------------------------------------------
    stream_analysis(
        image_bytes,
        filename=filename,
        mime=mime,
        lat=lat,
        lng=lng,
        user_id=args.user,
        base_url=args.url,
        raw_json=args.raw_json,
    )


if __name__ == "__main__":
    main()
