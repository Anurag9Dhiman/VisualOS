"""MCP server (stdio) — exposes the Lens OS pipeline as tools.

Run with:
    python -m src.mcp_server

Register in any MCP client, e.g. Claude Code (.claude/settings.json):
    {
      "mcpServers": {
        "lens-os": {
          "command": "python",
          "args": ["-m", "src.mcp_server"],
          "cwd": "/path/to/VisualOS"
        }
      }
    }

Or Cursor (~/.cursor/mcp.json), Zed (mcp_servers in settings.json),
VS Code Copilot (.vscode/mcp.json) — same structure, different file path.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from mcp.server.mcpserver import MCPServer  # noqa: E402

from src.contracts import LensInput  # noqa: E402
from src.orchestrator import run_pipeline  # noqa: E402

mcp = MCPServer(
    "lens-os",
    instructions=(
        "Lens OS is a visual intelligence agent. "
        "Point a camera at any building, monument, statue, or object to get "
        "identification, history, and live info in under 3 seconds. "
        "Always pass image_b64 as a base64-encoded JPEG/PNG/WebP. "
        "lat and lng improve accuracy but are optional."
    ),
)


@mcp.tool()
async def analyze_image(
    image_b64: str,
    mime_type: str = "image/jpeg",
    lat: float | None = None,
    lng: float | None = None,
    user_id: str = "anon",
    user_locale: str = "en-IN",
) -> dict:
    """Identify a building, monument, statue, or object from a photo.

    Args:
        image_b64: Base64-encoded image bytes (JPEG, PNG, or WebP).
        mime_type: MIME type of the image (default: image/jpeg).
        lat: GPS latitude of the photo location — improves accuracy.
        lng: GPS longitude of the photo location — improves accuracy.
        user_id: Stable identifier for the user (enables memory/personalisation).
        user_locale: BCP-47 locale for the response card (default: en-IN).

    Returns:
        dict with "card" (ResponseCard) and "session_id" for voice follow-up.
    """
    suffix = ".png" if mime_type == "image/png" else ".jpg"
    image_bytes = base64.b64decode(image_b64)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        inp = LensInput(
            image_path=tmp_path,
            lat=lat,
            lng=lng,
            user_id=user_id,
            user_locale=user_locale,
        )
        state = await run_pipeline(inp)
        card = state.get("response_card")
        if card is None:
            return {"error": "Pipeline produced no card", "errors": state.get("errors", [])}
        return {"card": card.model_dump(), "session_id": state.get("session_id")}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    mcp.run(transport="stdio")
