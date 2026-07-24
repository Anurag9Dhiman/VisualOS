"""Tests for Pydantic v2 contracts — no API calls needed."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.contracts import (
    FallbackCard,
    MemoryHit,
    MemoryResult,
    NormalCard,
    PersonalizedHook,
    SearchResult,
    SourceMix,
    VisionResult,
)

# ---------------------------------------------------------------------------
# VisionResult
# ---------------------------------------------------------------------------


def _valid_vision(**overrides) -> dict:
    base = {
        "entity_name": "Lalbagh West Gate",
        "entity_type": "monument",
        "confidence_level": "certain",
        "evidence": ["signage matches"],
        "failure_modes_checked": ["lighting checked"],
        "needs_fallback": False,
    }
    return {**base, **overrides}


def test_vision_valid():
    v = VisionResult(**_valid_vision())
    assert v.confidence_score == 0.95
    assert v.subject_name == "Lalbagh West Gate"


def test_vision_confidence_scores():
    levels = {
        "certain": 0.95,
        "fairly_sure": 0.80,
        "uncertain": 0.50,
        "guessing": 0.20,
    }
    for level, expected in levels.items():
        needs = level == "guessing"
        v = VisionResult(**_valid_vision(confidence_level=level, needs_fallback=needs))
        assert v.confidence_score == expected


def test_vision_guessing_requires_fallback():
    with pytest.raises(ValidationError, match="needs_fallback"):
        VisionResult(**_valid_vision(confidence_level="guessing", needs_fallback=False))


def test_vision_guessing_with_fallback_passes():
    v = VisionResult(**_valid_vision(confidence_level="guessing", needs_fallback=True))
    assert v.needs_fallback is True


def test_vision_evidence_required():
    with pytest.raises(ValidationError):
        VisionResult(**_valid_vision(evidence=[]))


def test_vision_failure_modes_required():
    with pytest.raises(ValidationError):
        VisionResult(**_valid_vision(failure_modes_checked=[]))


def test_vision_invalid_entity_type():
    with pytest.raises(ValidationError):
        VisionResult(**_valid_vision(entity_type="spaceship"))


def test_vision_invalid_confidence():
    with pytest.raises(ValidationError):
        VisionResult(**_valid_vision(confidence_level="very_sure"))


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


def _valid_search(**overrides) -> dict:
    base = {
        "research_plan": "Use Wikipedia first, then Tavily.",
        "tool_calls": [],
        "historical_facts": [{"fact": "Built in 1760", "source": "Wikipedia"}],
        "live_facts": [],
        "live_facts_skipped_reason": "No time-sensitive info for this historical site.",
    }
    return {**base, **overrides}


def test_search_valid():
    s = SearchResult(**_valid_search())
    assert s.tool_call_count == 0
    assert s.all_sources == ["Wikipedia"]


def test_search_budget_enforced():
    tool_calls = [
        {"tool": "wikipedia_summary", "input": {}, "justification": "j", "observation": "o"}
    ] * 4
    with pytest.raises(ValidationError, match="tool_calls"):
        SearchResult(**_valid_search(tool_calls=tool_calls))


def test_search_exactly_3_tool_calls_passes():
    tool_calls = [
        {"tool": "wikipedia_summary", "input": {}, "justification": "j", "observation": "o"}
    ] * 3
    s = SearchResult(**_valid_search(tool_calls=tool_calls))
    assert s.tool_call_count == 3


def test_search_live_facts_empty_needs_reason():
    with pytest.raises(ValidationError, match="live_facts_skipped_reason"):
        SearchResult(**_valid_search(live_facts=[], live_facts_skipped_reason=""))


def test_search_live_facts_present_no_reason_ok():
    s = SearchResult(
        **_valid_search(
            live_facts=[{"fact": "Open 9am-6pm", "source": "Tavily", "as_of": "2026-06"}],
            live_facts_skipped_reason="",
        )
    )
    assert len(s.live_facts) == 1
    assert "Tavily" in s.all_sources


def test_search_all_sources_combines():
    s = SearchResult(
        **_valid_search(
            historical_facts=[
                {"fact": "f1", "source": "Wikipedia"},
                {"fact": "f2", "source": "Wikidata"},
            ],
            live_facts=[{"fact": "f3", "source": "Tavily", "as_of": "now"}],
            live_facts_skipped_reason="",
        )
    )
    assert set(s.all_sources) == {"Wikipedia", "Wikidata", "Tavily"}


# ---------------------------------------------------------------------------
# NormalCard
# ---------------------------------------------------------------------------


def test_normal_card_valid():
    card = NormalCard(
        headline="This is Lalbagh.",
        body="Built in the 18th century.",
        personalized_hooks=[],
        citations=[],
        confidence_displayed="high",
        source_mix=SourceMix(used_vision=True, used_memory=False, used_search=True),
    )
    assert card.card_type == "normal"


def test_normal_card_max_3_hooks():
    hooks = [PersonalizedHook(fact=f"fact {i}", citation_tag=f"c{i}") for i in range(4)]
    with pytest.raises(ValidationError):
        NormalCard(
            headline="h",
            body="b",
            personalized_hooks=hooks,
            citations=[],
            confidence_displayed="high",
            source_mix=SourceMix(used_vision=True, used_memory=False, used_search=False),
        )


def test_normal_card_exactly_3_hooks_ok():
    hooks = [PersonalizedHook(fact=f"fact {i}", citation_tag=f"c{i}") for i in range(3)]
    card = NormalCard(
        headline="h",
        body="b",
        personalized_hooks=hooks,
        citations=[],
        confidence_displayed="high",
        source_mix=SourceMix(used_vision=True, used_memory=False, used_search=False),
    )
    assert len(card.personalized_hooks) == 3


# ---------------------------------------------------------------------------
# FallbackCard
# ---------------------------------------------------------------------------


def test_fallback_card_valid():
    card = FallbackCard(
        headline="Not sure what this is.",
        observation="Partially occluded statue.",
        suggestion="Try a clearer angle.",
    )
    assert card.card_type == "fallback"
    assert card.cost_usd_total == 0.0


# ---------------------------------------------------------------------------
# MemoryResult
# ---------------------------------------------------------------------------


def test_memory_result_empty_interests():
    m = MemoryResult(hits=[], user_id="u1")
    assert m.user_interests_snapshot == {}


def test_memory_hit_roundtrip():
    h = MemoryHit(
        interaction_id="abc",
        subject_name="Red Fort",
        summary="A Mughal monument in Delhi.",
        timestamp=datetime(2026, 6, 1),
        similarity_score=0.92,
    )
    assert h.similarity_score == 0.92
