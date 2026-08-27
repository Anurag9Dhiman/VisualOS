"""Lens OS — macOS menu bar app.

Always-on entry point to Lens OS. Sits in the menu bar and lets you
point at anything on screen, in your clipboard, or from a file and get
instant AI identification — without touching the terminal.

Usage:
  python -m src.mac_menubar            # uses default server at :8765
  python -m src.mac_menubar --url http://192.168.1.10:8765

Requirements (in addition to the base requirements.txt):
  pip install rumps

The Lens OS server must be running before scanning:
  uvicorn src.server:app --port 8765

Global keyboard shortcut:
  macOS does not grant apps global hotkeys without Accessibility permission.
  To bind ⌘⇧L: System Settings → Keyboard → Keyboard Shortcuts →
  App Shortcuts → add "Scan Screen Region" for this app.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import tempfile
import threading
from pathlib import Path

import httpx
import rumps
from PIL import Image, ImageGrab

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _choose_file() -> str | None:
    """Open a native macOS file-chooser via AppleScript. Returns POSIX path or None."""
    script = (
        'POSIX path of (choose file with prompt "Select an image to analyse" '
        'of type {"public.jpeg", "public.png", "org.webmproject.webp"})'
    )
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
    )
    path = result.stdout.strip()
    return path if path else None


def _capture_screen() -> bytes | None:
    """Interactive region capture (like Cmd+Shift+4). Returns JPEG bytes or None."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    result = subprocess.run(
        ["screencapture", "-i", "-s", "-t", "jpg", tmp_path],
        capture_output=True,
    )
    data = Path(tmp_path).read_bytes() if Path(tmp_path).exists() else b""
    Path(tmp_path).unlink(missing_ok=True)
    if result.returncode != 0 or not data:
        return None
    return data


def _capture_clipboard() -> bytes | None:
    """Grab an image from the system clipboard. Returns JPEG bytes or None."""
    img = ImageGrab.grabclipboard()
    if not isinstance(img, Image.Image):
        return None
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _load_file(path: str) -> bytes | None:
    """Load an image file from disk. Returns raw bytes or None."""
    p = Path(path)
    if not p.exists() or p.suffix.lower() not in _ALLOWED_SUFFIXES:
        return None
    return p.read_bytes()


def _mime_for(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")


def _ip_location() -> tuple[float | None, float | None]:
    """Best-effort city-level location from public IP."""
    try:
        resp = httpx.get("https://ipapi.co/json/", timeout=4.0)
        data = resp.json()
        return float(data["latitude"]), float(data["longitude"])
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Menu bar app
# ---------------------------------------------------------------------------


class LensMenuBar(rumps.App):
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._last_session_id: str | None = None
        self._last_entity: str | None = None
        self._scanning = False

        # Build menu
        self._item_last = rumps.MenuItem("Last scan: —")
        self._item_last.set_callback(None)

        self._item_copy = rumps.MenuItem("Copy Session ID", callback=self._copy_session_id)
        self._item_copy.set_callback(None)  # disabled until first scan

        self._item_server = rumps.MenuItem(f"Server: {base_url}")
        self._item_server.set_callback(None)

        super().__init__(
            "🔍",
            menu=[
                rumps.MenuItem("Scan Screen Region", callback=self._scan_screen, key="l"),
                rumps.MenuItem("Scan from Clipboard", callback=self._scan_clipboard),
                rumps.MenuItem("Analyze File…", callback=self._scan_file),
                None,  # separator
                self._item_last,
                self._item_copy,
                None,
                self._item_server,
                rumps.MenuItem("Change Server…", callback=self._change_server),
            ],
            quit_button="Quit Lens OS",
        )

    # ------------------------------------------------------------------
    # Capture entry points (all called on main thread by rumps)
    # ------------------------------------------------------------------

    def _scan_screen(self, _sender) -> None:
        if self._scanning:
            return
        image_bytes = _capture_screen()
        if not image_bytes:
            rumps.notification("Lens OS", "", "Screen capture cancelled.", sound=False)
            return
        self._start_scan(image_bytes, "capture.jpg", "image/jpeg")

    def _scan_clipboard(self, _sender) -> None:
        if self._scanning:
            return
        image_bytes = _capture_clipboard()
        if not image_bytes:
            rumps.notification(
                "Lens OS", "", "No image on clipboard. Copy an image first.", sound=False
            )
            return
        self._start_scan(image_bytes, "clipboard.jpg", "image/jpeg")

    def _scan_file(self, _sender) -> None:
        if self._scanning:
            return
        path = _choose_file()
        if not path:
            return
        image_bytes = _load_file(path)
        if not image_bytes:
            rumps.notification("Lens OS", "", f"Could not load: {Path(path).name}", sound=False)
            return
        self._start_scan(image_bytes, Path(path).name, _mime_for(path))

    # ------------------------------------------------------------------
    # Background scan
    # ------------------------------------------------------------------

    def _start_scan(self, image_bytes: bytes, filename: str, mime: str) -> None:
        self._scanning = True
        self.title = "⏳"
        threading.Thread(
            target=self._run_scan,
            args=(image_bytes, filename, mime),
            daemon=True,
        ).start()

    def _run_scan(self, image_bytes: bytes, filename: str, mime: str) -> None:
        try:
            lat, lng = _ip_location()
            data: dict = {"user_id": "menubar-user", "user_locale": "en-IN"}
            if lat is not None:
                data["lat"] = str(lat)
            if lng is not None:
                data["lng"] = str(lng)

            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{self._base_url}/analyze",
                    data=data,
                    files={"image": (filename, image_bytes, mime)},
                )

            if resp.status_code != 200:
                self._on_error(f"Server error {resp.status_code}")
                return

            result = resp.json()
            card = result.get("card", {})
            session_id = result.get("session_id")
            headline = card.get("headline", "Unknown entity")
            body = card.get("body", "")
            card_type = card.get("card_type", "normal")

            self._on_success(headline, body, session_id, card_type)

        except httpx.ConnectError:
            self._on_error(f"Cannot reach server at {self._base_url}")
        except Exception as exc:
            self._on_error(str(exc))

    def _on_success(self, headline: str, body: str, session_id: str | None, card_type: str) -> None:
        self._last_entity = headline
        self._last_session_id = session_id
        self._scanning = False

        # Truncate headline for menu display
        label = headline if len(headline) <= 50 else headline[:47] + "…"
        self._item_last.title = f"Last scan: {label}"

        if session_id:
            self._item_copy.set_callback(self._copy_session_id)
            self._item_copy.title = "Copy Session ID"
        else:
            self._item_copy.set_callback(None)

        # Notification
        subtitle = "Identified" if card_type == "normal" else "Low confidence"
        preview = body[:120] + "…" if len(body) > 120 else body
        rumps.notification("Lens OS", subtitle, f"{headline}\n{preview}", sound=True)

        self.title = "✅"
        # Reset icon after 3 s
        threading.Timer(3.0, self._reset_icon).start()

    def _on_error(self, message: str) -> None:
        self._scanning = False
        rumps.notification("Lens OS", "Scan failed", message, sound=False)
        self.title = "❌"
        threading.Timer(3.0, self._reset_icon).start()

    def _reset_icon(self) -> None:
        self.title = "🔍"

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------

    def _copy_session_id(self, _sender) -> None:
        if not self._last_session_id:
            return
        subprocess.run(
            ["pbcopy"],
            input=self._last_session_id.encode(),
            check=False,
        )
        rumps.notification(
            "Lens OS",
            "Copied",
            "Session ID copied to clipboard.\nPaste into VoiceOS as scan_session_id.",
            sound=False,
        )

    def _change_server(self, _sender) -> None:
        response = rumps.Window(
            message="Enter the Lens OS server URL:",
            title="Change Server",
            default_text=self._base_url,
            ok="Save",
            cancel="Cancel",
            dimensions=(300, 20),
        ).run()
        if response.clicked and response.text.strip():
            new_url = response.text.strip().rstrip("/")
            self._base_url = new_url
            self._item_server.title = f"Server: {new_url}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Lens OS menu bar app")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8765",
        help="Lens OS server URL (default: http://127.0.0.1:8765)",
    )
    args = parser.parse_args()

    app = LensMenuBar(base_url=args.url)
    app.run()


if __name__ == "__main__":
    main()
