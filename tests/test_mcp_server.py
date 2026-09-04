"""Tests for the MCP server tool definitions — no API keys needed.

These tests validate the tool's input parsing and pipeline integration
without starting the MCP stdio transport itself.
"""

from __future__ import annotations

import base64
import io
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

pytest.importorskip("mcp", reason="mcp package not installed — skipping MCP server tests")

from src.contracts import NormalCard  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jpeg_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(200, 100, 50)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _normal_card() -> NormalCard:
    return NormalCard(
        headline="Lalbagh Gate",
        body="Colonial-era gate in Bangalore.",
        personalized_hooks=[],
        citations=[],
        confidence_displayed="high",
        source_mix={"used_vision": True, "used_memory": False, "used_search": False},
        cost_usd_total=0.0005,
        latency_ms=600,
    )


# ---------------------------------------------------------------------------
# analyze_image tool
# ---------------------------------------------------------------------------


async def test_analyze_image_returns_card_dict():
    from src.mcp_server import analyze_image

    mock_state = {"response_card": _normal_card(), "errors": []}
    with patch("src.mcp_server.run_pipeline", new_callable=AsyncMock, return_value=mock_state):
        result = await analyze_image(image_b64=_jpeg_b64())

    assert "card" in result
    assert result["card"]["headline"] == "Lalbagh Gate"
    assert result["card"]["card_type"] == "normal"


async def test_analyze_image_passes_gps():
    from src.mcp_server import analyze_image

    mock_state = {"response_card": _normal_card(), "errors": []}
    with patch(
        "src.mcp_server.run_pipeline", new_callable=AsyncMock, return_value=mock_state
    ) as mock:
        await analyze_image(image_b64=_jpeg_b64(), lat=12.97, lng=77.59)

    inp = mock.call_args[0][0]
    assert inp.lat == pytest.approx(12.97)
    assert inp.lng == pytest.approx(77.59)


async def test_analyze_image_returns_error_when_no_card():
    from src.mcp_server import analyze_image

    mock_state = {"response_card": None, "errors": ["vision failed"]}
    with patch("src.mcp_server.run_pipeline", new_callable=AsyncMock, return_value=mock_state):
        result = await analyze_image(image_b64=_jpeg_b64())

    assert "error" in result
    assert result["errors"] == ["vision failed"]


async def test_analyze_image_accepts_png_mime():
    from src.mcp_server import analyze_image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode()

    mock_state = {"response_card": _normal_card(), "errors": []}
    with patch(
        "src.mcp_server.run_pipeline", new_callable=AsyncMock, return_value=mock_state
    ) as mock:
        result = await analyze_image(image_b64=png_b64, mime_type="image/png")

    inp = mock.call_args[0][0]
    assert inp.image_path.endswith(".png")
    assert "card" in result
