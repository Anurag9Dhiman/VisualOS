"""Tests for Layer-1 in-context memory (memory_l1.py)."""

from __future__ import annotations

import time

import pytest

from src import memory_l1


@pytest.fixture(autouse=True)
def _reset():
    """Clear L1 state before each test so tests are independent."""
    memory_l1.clear()
    yield
    memory_l1.clear()


def test_get_recent_context_empty_for_new_user():
    assert memory_l1.get_recent_context("new-user") == ""


def test_record_and_retrieve_single_scan():
    memory_l1.record_scan("u1", "India Gate", "A war memorial in New Delhi.")
    ctx = memory_l1.get_recent_context("u1")
    assert "India Gate" in ctx


def test_recent_context_lists_multiple_scans():
    memory_l1.record_scan("u1", "India Gate", "Monument.")
    memory_l1.record_scan("u1", "Qutub Minar", "Tower.")
    ctx = memory_l1.get_recent_context("u1")
    assert "India Gate" in ctx
    assert "Qutub Minar" in ctx


def test_recent_context_only_last_three_names():
    for name in ["A", "B", "C", "D", "E"]:
        memory_l1.record_scan("u1", name, "summary")
    ctx = memory_l1.get_recent_context("u1")
    # Only the last 3 scans appear in the 1-sentence summary
    assert "C" in ctx or "D" in ctx or "E" in ctx
    # "A" and "B" are too old to be in the last-3 window
    assert "A" not in ctx
    assert "B" not in ctx


def test_users_are_isolated():
    memory_l1.record_scan("alice", "India Gate", "monument")
    memory_l1.record_scan("bob", "Eiffel Tower", "tower")
    assert "India Gate" in memory_l1.get_recent_context("alice")
    assert "Eiffel Tower" not in memory_l1.get_recent_context("alice")
    assert "Eiffel Tower" in memory_l1.get_recent_context("bob")
    assert "India Gate" not in memory_l1.get_recent_context("bob")


def test_clear_single_user():
    memory_l1.record_scan("u1", "India Gate", "monument")
    memory_l1.record_scan("u2", "Colosseum", "amphitheatre")
    memory_l1.clear("u1")
    assert memory_l1.get_recent_context("u1") == ""
    assert "Colosseum" in memory_l1.get_recent_context("u2")


def test_clear_all_users():
    memory_l1.record_scan("u1", "India Gate", "monument")
    memory_l1.record_scan("u2", "Colosseum", "amphitheatre")
    memory_l1.clear()
    assert memory_l1.get_recent_context("u1") == ""
    assert memory_l1.get_recent_context("u2") == ""


def test_entries_expire_after_ttl(monkeypatch):
    """Entries older than _TTL_S should be treated as expired."""
    memory_l1.record_scan("u1", "Old Entity", "stale summary")

    # Monkey-patch time.monotonic to return a time far in the future
    future = time.monotonic() + memory_l1._TTL_S + 1
    monkeypatch.setattr("src.memory_l1.time.monotonic", lambda: future)

    ctx = memory_l1.get_recent_context("u1")
    assert ctx == ""  # expired entry filtered out


def test_maxlen_deque_drops_oldest():
    """When more than _MAX_ENTRIES scans are recorded, the oldest is dropped."""
    for i in range(memory_l1._MAX_ENTRIES + 2):
        memory_l1.record_scan("u1", f"Entity{i}", "summary")

    # The store uses a deque with maxlen — len should not exceed _MAX_ENTRIES
    assert len(memory_l1._store["u1"]) == memory_l1._MAX_ENTRIES
