"""Tests for entity-type routing — route_node, _should_search_after_route, tool priorities."""

from __future__ import annotations

import pytest

from src.contracts import LensInput, VisionResult
from src.orchestrator import (
    _ENTITY_ROUTE,
    _ROUTE_TO_TOOL_PRIORITY,
    LensState,
    _should_search_after_route,
    route_node,
)
from src.prompts import GeoPoint, build_search_user_message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vision(
    entity_type: str = "building",
    confidence_level: str = "certain",
    needs_fallback: bool = False,
) -> VisionResult:
    return VisionResult(
        entity_name="Test Entity",
        entity_type=entity_type,  # type: ignore[arg-type]
        confidence_level=confidence_level,  # type: ignore[arg-type]
        evidence=["some evidence"],
        alternatives=[],
        failure_modes_checked=["lighting"],
        needs_fallback=needs_fallback,
    )


def _make_state(vision: VisionResult | None) -> LensState:
    return {
        "input": LensInput(image_path="fake.jpg"),
        "image_b64": "",
        "vision_result": vision,
        "memory_result": None,
        "search_result": None,
        "response_card": None,
        "cost_log": [],
        "errors": [],
        "_start_time": 0.0,
        "_cache_key": "",
        "search_route": "default",
    }


# ---------------------------------------------------------------------------
# route_node — entity type dispatch
# ---------------------------------------------------------------------------


def test_building_routes_osm_first() -> None:
    result = route_node(_make_state(_make_vision(entity_type="building")))
    assert result["search_route"] == "osm_first"


def test_monument_routes_wikidata_first() -> None:
    result = route_node(_make_state(_make_vision(entity_type="monument")))
    assert result["search_route"] == "wikidata_first"


def test_statue_routes_wikidata_first() -> None:
    result = route_node(_make_state(_make_vision(entity_type="statue")))
    assert result["search_route"] == "wikidata_first"


def test_object_routes_skip() -> None:
    result = route_node(_make_state(_make_vision(entity_type="object")))
    assert result["search_route"] == "skip"


def test_unknown_routes_default() -> None:
    result = route_node(_make_state(_make_vision(entity_type="unknown")))
    assert result["search_route"] == "default"


# ---------------------------------------------------------------------------
# route_node — confidence gate takes precedence over entity type
# ---------------------------------------------------------------------------


def test_guessing_routes_skip_regardless_of_entity_type() -> None:
    vision = _make_vision(entity_type="building", confidence_level="guessing", needs_fallback=True)
    result = route_node(_make_state(vision))
    assert result["search_route"] == "skip"


def test_needs_fallback_routes_skip() -> None:
    vision = _make_vision(entity_type="monument", confidence_level="uncertain", needs_fallback=True)
    result = route_node(_make_state(vision))
    assert result["search_route"] == "skip"


def test_none_vision_routes_skip() -> None:
    result = route_node(_make_state(None))
    assert result["search_route"] == "skip"


def test_uncertain_confidence_still_searches_if_not_fallback() -> None:
    vision = _make_vision(entity_type="statue", confidence_level="uncertain", needs_fallback=False)
    result = route_node(_make_state(vision))
    assert result["search_route"] == "wikidata_first"


# ---------------------------------------------------------------------------
# _should_search_after_route
# ---------------------------------------------------------------------------


def test_skip_route_returns_fuse() -> None:
    state = _make_state(_make_vision())
    state["search_route"] = "skip"
    assert _should_search_after_route(state) == "fuse"


@pytest.mark.parametrize("route", ["default", "osm_first", "wikidata_first"])
def test_non_skip_routes_return_search(route: str) -> None:
    state = _make_state(_make_vision())
    state["search_route"] = route
    assert _should_search_after_route(state) == "search"


# ---------------------------------------------------------------------------
# _ROUTE_TO_TOOL_PRIORITY — priority ordering
# ---------------------------------------------------------------------------


def test_osm_first_priority_starts_with_osm_nearby() -> None:
    assert _ROUTE_TO_TOOL_PRIORITY["osm_first"][0] == "osm_nearby"


def test_wikidata_first_priority_starts_with_wikidata_query() -> None:
    assert _ROUTE_TO_TOOL_PRIORITY["wikidata_first"][0] == "wikidata_query"


def test_default_priority_starts_with_wikipedia_summary() -> None:
    assert _ROUTE_TO_TOOL_PRIORITY["default"][0] == "wikipedia_summary"


def test_all_routes_have_three_tools() -> None:
    for route, tools in _ROUTE_TO_TOOL_PRIORITY.items():
        assert len(tools) == 3, f"route={route!r} has {len(tools)} tools, expected 3"


# ---------------------------------------------------------------------------
# build_search_user_message — tool_priority injection
# ---------------------------------------------------------------------------


def test_search_user_message_includes_tool_priority() -> None:
    msg = build_search_user_message(
        entity_name="Test Building",
        entity_type="building",
        vision_confidence_level="certain",
        location=GeoPoint(lat=12.9, lng=77.5),
        user_interests=["architecture"],
        tool_priority=["osm_nearby", "wikipedia_summary", "tavily_search"],
    )
    assert "tool_priority" in msg
    assert "osm_nearby" in msg


def test_search_user_message_omits_priority_when_none() -> None:
    msg = build_search_user_message(
        entity_name="Test Statue",
        entity_type="statue",
        vision_confidence_level="fairly_sure",
        location=GeoPoint(lat=28.6, lng=77.2),
        user_interests=[],
    )
    assert "tool_priority" not in msg
    assert "Apply the reasoning protocol" in msg


def test_entity_route_covers_all_entity_types() -> None:
    """Every entity_type in VisionResult must have a route entry."""
    from typing import get_args

    from src.contracts import VisionResult

    entity_types = set(get_args(VisionResult.model_fields["entity_type"].annotation))
    assert entity_types == set(_ENTITY_ROUTE.keys()), (
        f"Missing routes for: {entity_types - set(_ENTITY_ROUTE.keys())}"
    )
