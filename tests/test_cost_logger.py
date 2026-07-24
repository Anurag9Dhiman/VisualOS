"""Tests for cost computation — no API calls needed."""

from __future__ import annotations

import pytest

from src.cost_logger import compute_cost, log_cost


def test_gemini_flash_cost():
    # 1000 input + 1000 output at gemini-2.0-flash rates
    cost = compute_cost("gemini-2.0-flash", 1000, 1000)
    assert cost == pytest.approx(0.000075 + 0.0003, rel=1e-6)


def test_embedding_cost_output_is_free():
    # text-embedding-004 has no output cost
    cost = compute_cost("text-embedding-004", 1000, 0)
    assert cost == pytest.approx(0.000025, rel=1e-6)
    # even if output_tokens passed accidentally, output price is 0
    cost_with_output = compute_cost("text-embedding-004", 1000, 500)
    assert cost == pytest.approx(cost_with_output, rel=1e-9)


def test_unknown_model_uses_fallback():
    cost_unknown = compute_cost("some-unknown-model", 1000, 1000)
    expected = (1000 / 1000) * 0.001 + (1000 / 1000) * 0.002
    assert cost_unknown == pytest.approx(expected, rel=1e-6)


def test_zero_tokens():
    assert compute_cost("gemini-2.0-flash", 0, 0) == 0.0


def test_log_cost_returns_entry():
    entry = log_cost("vision", "gemini-2.0-flash", 500, 200)
    assert entry.agent == "vision"
    assert entry.model == "gemini-2.0-flash"
    assert entry.input_tokens == 500
    assert entry.output_tokens == 200
    assert entry.cost_usd == pytest.approx(compute_cost("gemini-2.0-flash", 500, 200), rel=1e-9)
