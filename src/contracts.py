"""Pydantic v2 contracts for every agent I/O boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

ConfidenceLevel = Literal["certain", "fairly_sure", "uncertain", "guessing"]

_CONFIDENCE_TO_FLOAT = {
    "certain": 0.95,
    "fairly_sure": 0.80,
    "uncertain": 0.50,
    "guessing": 0.20,
}


# ---------------------------------------------------------------------------
# Orchestrator input / state
# ---------------------------------------------------------------------------

class LensInput(BaseModel):
    image_path: str
    lat: float | None = None
    lng: float | None = None
    user_id: str = "anon"
    user_locale: str = "en-IN"


class CostEntry(BaseModel):
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Vision Agent
# ---------------------------------------------------------------------------

class VisionResult(BaseModel):
    entity_name: str
    entity_type: Literal["building", "monument", "statue", "object", "unknown"]
    confidence_level: ConfidenceLevel
    evidence: list[str] = Field(min_length=1)
    alternatives: list[str] = []
    failure_modes_checked: list[str] = Field(min_length=1)
    needs_fallback: bool

    @model_validator(mode="after")
    def _guessing_requires_fallback(self) -> "VisionResult":
        if self.confidence_level == "guessing" and not self.needs_fallback:
            raise ValueError("needs_fallback must be True when confidence_level is 'guessing'")
        return self

    @property
    def confidence_score(self) -> float:
        return _CONFIDENCE_TO_FLOAT[self.confidence_level]

    @property
    def subject_name(self) -> str:
        return self.entity_name


# ---------------------------------------------------------------------------
# Memory Agent
# ---------------------------------------------------------------------------

class MemoryHit(BaseModel):
    interaction_id: str
    subject_name: str
    summary: str
    timestamp: datetime
    similarity_score: float


class MemoryResult(BaseModel):
    hits: list[MemoryHit]
    user_id: str
    user_interests_snapshot: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Search Agent
# ---------------------------------------------------------------------------

class HistoricalFact(BaseModel):
    fact: str
    source: str


class LiveFact(BaseModel):
    fact: str
    source: str
    as_of: str


class ToolCallRecord(BaseModel):
    tool: str
    input: dict
    justification: str
    observation: str


class SearchResult(BaseModel):
    research_plan: str
    tool_calls: list[ToolCallRecord]
    historical_facts: list[HistoricalFact]
    live_facts: list[LiveFact]
    live_facts_skipped_reason: str = ""
    identification_concerns: list[str] = []
    nearby_context: str = ""

    @model_validator(mode="after")
    def _validate_budget_and_skipped(self) -> "SearchResult":
        if len(self.tool_calls) > 3:
            raise ValueError("tool_calls must not exceed 3 (budget limit)")
        if not self.live_facts and not self.live_facts_skipped_reason:
            raise ValueError("live_facts_skipped_reason is required when live_facts is empty")
        return self

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def all_sources(self) -> list[str]:
        return (
            [f.source for f in self.historical_facts]
            + [f.source for f in self.live_facts]
        )


# ---------------------------------------------------------------------------
# Fusion output — union card types
# ---------------------------------------------------------------------------

class PersonalizedHook(BaseModel):
    fact: str
    citation_tag: str


class Citation(BaseModel):
    id: str
    source_name: str
    url: str
    as_of: str | None = None


class SourceMix(BaseModel):
    used_vision: bool
    used_memory: bool
    used_search: bool


class NormalCard(BaseModel):
    card_type: Literal["normal"] = "normal"
    headline: str
    body: str
    personalized_hooks: Annotated[list[PersonalizedHook], Field(max_length=3)] = []
    citations: list[Citation]
    confidence_displayed: Literal["high", "hedged"]
    source_mix: SourceMix
    cost_usd_total: float = 0.0
    latency_ms: int = 0


class FallbackCard(BaseModel):
    card_type: Literal["fallback"] = "fallback"
    headline: str
    observation: str
    suggestion: str
    cost_usd_total: float = 0.0
    latency_ms: int = 0


ResponseCard = NormalCard | FallbackCard


# ---------------------------------------------------------------------------
# Tool result types
# ---------------------------------------------------------------------------

class ToolError(Exception):
    def __init__(self, tool: str, message: str) -> None:
        self.tool = tool
        super().__init__(f"{tool}: {message}")


class WikipediaResult(BaseModel):
    title: str
    extract: str
    url: str


class WikidataResult(BaseModel):
    entity_id: str
    label: str
    facts: dict[str, str]
    url: str


class TavilyResult(BaseModel):
    query: str
    results: list[dict]


class OSMResult(BaseModel):
    name: str | None
    address: str | None
    opening_hours: str | None
    wheelchair: str | None
