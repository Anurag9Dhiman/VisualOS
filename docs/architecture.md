# Lens OS — Architecture

Phase 0 proof-of-concept. The goal is to validate the full loop in Python before writing any production iOS code.

---

## 1. System overview

```
Image + GPS
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ Orchestrator (LangGraph)                            │
│                                                     │
│  plan ──► cache_check ──► vision_memory ──► fuse ──► write_memory
│                │                   │
│                │ (HIT)             ▼
│                │           confidence_gate
│                │               │       │
│                │           search   (skip)
│                │               │       │
│                └───────────────▼───────┘
│                              fuse
└─────────────────────────────────────────────────────┘
     │
     ▼
ResponseCard (NormalCard | FallbackCard)
```

Three specialist agents run in parallel inside `vision_memory`:
- **Vision** — identifies the entity from the image
- **Memory** — retrieves prior interactions and user interests

Search runs after, gated by Vision's confidence. Fusion composes the final card.

---

## 2. LangGraph graph

Defined in `src/orchestrator.py`. Nodes and edges:

| Node | Type | What it does |
|---|---|---|
| `plan` | sync | Reads image bytes, base64-encodes, computes cache key |
| `cache_check` | async | Looks up `(image_hash + location)` in SQLite cache |
| `vision_memory` | async | Runs Vision + Memory in `asyncio.gather` |
| `search` | async | ReAct search, 3-tool budget |
| `fuse` | async | Gemini composes NormalCard or FallbackCard, writes to cache |
| `write_memory` | async | Fire-and-forget: embeds entity name, writes interaction to DB |

Conditional edges:
- `cache_check → done` if cache HIT (skips all agents)
- `vision_memory → fuse` if Vision returns `guessing` or `needs_fallback=True` (confidence gate)
- `vision_memory → search` otherwise

---

## 3. Agents

### Vision (`src/agents/vision.py`)

- **Model**: `gemini-2.0-flash` with vision input
- **Input**: base64 JPEG (preprocessed to ≤1024px, quality 85)
- **Prompt**: 5-step reasoning — identify → cross-check → alternatives → failure modes → commit confidence
- **Output**: `VisionResult` (see §5)
- **Timeout**: 800ms hard cutoff via `asyncio.wait_for`

### Memory (`src/agents/memory.py`)

- **Model**: `text-embedding-004` (768-dim vectors)
- **What it does**: embeds the entity name, cosine-searches past interactions in SQLite, returns top-5 hits above 0.75 similarity threshold + user interest snapshot
- **Timeout**: 400ms (200ms embed + 200ms DB search)

### Search (`src/agents/search.py`)

- **Model**: `gemini-2.0-flash`
- **Pattern**: ReAct (reason → tool → observe), hard 3-call budget
- **Tools**: `wikipedia_summary`, `wikidata_query`, `tavily_search`, `osm_nearby`
- **Output**: `SearchResult` with historical facts, live facts, tool call trace
- **Timeout**: 800ms

### Fusion (`src/fusion.py`)

- **Model**: `gemini-2.0-flash`
- **Input**: all three agent outputs + user locale
- **Output**: `NormalCard` (headline + body + personalised hooks + citations) or `FallbackCard`
- **Two modes**: `run_fusion` (blocking, JSON output) and `stream_fusion` (async generator, token streaming)
- **Timeout**: 800ms for blocking mode

---

## 4. Pipeline features

### Confidence gate
If Vision returns `confidence_level="guessing"` or `needs_fallback=True`, Search is skipped entirely. Fusion still runs and returns a FallbackCard.

### Image preprocessing
Every image is resized to max 1024px and recompressed to JPEG quality 85 before being sent to Vision. Reduces payload by ~90% on typical phone photos. Implemented in `src/agents/vision._preprocess_image`.

### Response cache (`src/cache.py`)
Cache key = `SHA256(preprocessed_image_bytes + rounded_lat + rounded_lng)` where lat/lng are rounded to 3 decimal places (~100m precision). TTL is 24 hours. Hit → all agents skipped, cached card returned directly.

### Rate limiter (`src/rate_limiter.py`)
Sliding 60-second window per model. Default limits: `gemini-2.0-flash` → 15 RPM (Gemini free tier), `text-embedding-004` → 1500 RPM. `acquire(model)` releases the lock while sleeping so parallel Vision + Memory calls don't serialise. Override with `set_limit(model, rpm)`.

### User profiling (`src/db.py`)
Exponential-decay interest scoring: each scan of an `entity_type` adds 1.0 to its score; existing scores decay by 0.9 on each new interaction (`new_score = old_score * 0.9 + 1.0`). Top-10 interests are passed to Search to prioritise which facts to surface.

### Streaming output
`stream_fusion` is an async generator that yields `(chunk: str, None)` for each token and `(full_text, card)` as the final item. Used by `stream_pipeline` in the orchestrator and by the CLI by default.

---

## 5. Data contracts (`src/contracts.py`)

All agent I/O uses Pydantic v2. Raw dicts are never passed between agents.

```
LensInput
  image_path, lat, lng, user_id, user_locale

VisionResult
  entity_name, entity_type, confidence_level, evidence[],
  alternatives[], failure_modes_checked[], needs_fallback
  → confidence_score (property, float)

MemoryResult
  hits: MemoryHit[], user_id, user_interests_snapshot: dict[str, float]

SearchResult
  research_plan, tool_calls[] (≤3), historical_facts[],
  live_facts[], live_facts_skipped_reason (required if live_facts empty),
  identification_concerns[], nearby_context

NormalCard  (card_type="normal")
  headline, body, personalized_hooks[] (≤3), citations[],
  confidence_displayed, source_mix, cost_usd_total, latency_ms

FallbackCard  (card_type="fallback")
  headline, observation, suggestion, cost_usd_total, latency_ms

ResponseCard = NormalCard | FallbackCard
```

Key validation rules enforced by Pydantic:
- `confidence_level="guessing"` requires `needs_fallback=True`
- `evidence` and `failure_modes_checked` must be non-empty
- `tool_calls` length ≤ 3 (budget limit)
- `live_facts_skipped_reason` required when `live_facts` is empty
- `personalized_hooks` length ≤ 3

---

## 6. Storage (`src/db.py`)

SQLite database at `lens_memory.db`. Three tables:

| Table | Purpose |
|---|---|
| `interactions` | User scan history: subject name, summary, 768-dim embedding blob, expires_at |
| `cost_log` | Every LLM call: agent, model, input_tokens, output_tokens, cost_usd |
| `user_interests` | Decayed interest scores per (user_id, interest) |
| `response_cache` | Cached response cards, keyed by image+location hash (in `src/cache.py`) |

Interactions expire after 30 days (privacy requirement). Embeddings stored as packed float32 blobs (`struct.pack`). Cosine similarity computed in Python at query time (no vector extension needed for Phase 0 scale).

---

## 7. Cost logging

Every LLM call must go through `log_cost(agent, model, in_tokens, out_tokens)` in `src/cost_logger.py`. Returns a `CostEntry` and logs a structured JSON line:

```json
{"event": "llm_cost", "agent": "vision", "model": "gemini-2.0-flash",
 "input_tokens": 312, "output_tokens": 48, "cost_usd": 0.000038}
```

Prices (per 1000 tokens, input / output):

| Model | Input | Output |
|---|---|---|
| `gemini-2.0-flash` | $0.000075 | $0.0003 |
| `text-embedding-004` | $0.000025 | $0.00 |

---

## 8. Prompts (`src/prompts.py`)

Three immutable system prompts. **Never modify without explicit human review** — changes break prompt caching and invalidate eval baselines.

| Prompt | Key design |
|---|---|
| `VISION_SYSTEM_PROMPT` | Mandatory 5-step chain-of-thought. Structured JSON output with `failure_modes_checked` required. |
| `SEARCH_SYSTEM_PROMPT` | ReAct with tool descriptions and 3-call budget stated explicitly. `live_facts_skipped_reason` required when no live facts. |
| `FUSION_SYSTEM_PROMPT` | Confidence propagation rules (CASE A/B/C). Personalisation rules (use interests to pick facts, don't reveal them). |

---

## 9. Latency budget

Overall hard timeout: **2.5s** (enforced in `run_pipeline` via `asyncio.wait_for`).

```
Vision (800ms) ─┐
                 ├─► Fusion (800ms) ─► card
Memory (400ms) ─┘
                        │
Search (800ms) ─────────┘ (only if Vision confidence ≥ uncertain)
```

Vision and Memory run in parallel. Search only runs if the confidence gate passes. At P50 the pipeline targets <3s end-to-end including network latency.

---

## 10. File map

```
src/
  main.py           CLI entry point — streaming by default, --json for raw output
  orchestrator.py   LangGraph graph definition + run_pipeline / stream_pipeline
  contracts.py      All Pydantic v2 models
  prompts.py        System prompts (immutable) + user message builders
  fusion.py         Fusion step — run_fusion (blocking) + stream_fusion (streaming)
  db.py             SQLite helpers — interactions, cost_log, user_interests
  cache.py          Response cache — init_cache, make_cache_key, cache_get, cache_set
  cost_logger.py    log_cost — every LLM call goes through here
  rate_limiter.py   Sliding-window rate limiter per model
  inspector.py      Streamlit web inspector (dev tool)
  agents/
    vision.py       Vision agent + _preprocess_image
    memory.py       Memory agent — embed + cosine search + interests
    search.py       Search agent — ReAct loop + tool dispatch
  tools/
    wikipedia_client.py   MediaWiki REST API
    wikidata_client.py    Wikidata SPARQL endpoint
    tavily_client.py      Tavily Search API (requires TAVILY_API_KEY)
    osm_client.py         Overpass API (OpenStreetMap)
tests/
  conftest.py         tmp_db fixture (fresh SQLite per test), rate_limiter reset
  test_contracts.py   Pydantic validation rules
  test_cost_logger.py Cost computation per model
  test_db.py          SQLite write/search/TTL/cosine similarity
  test_rate_limiter.py Sliding-window slot tracking, sleep behaviour
  test_agents.py      Vision/Memory/Search/Fusion with mocked Gemini client
  test_tools.py       Wikipedia/Wikidata/Tavily/OSM with mocked httpx
  test_cache.py       Cache key, get/set/TTL/not-initialised guard
  test_fusion.py      _parse_card, stream_fusion chunks and error path
```

---

## 11. Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini models + text-embedding-004 |
| `TAVILY_API_KEY` | No | Live web search (Search agent falls back gracefully without it) |
| `LANGCHAIN_TRACING_V2` | No | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | LangSmith project name (e.g. `lens-os-phase0`) |

Copy `.env.example` to `.env` and fill in `GOOGLE_API_KEY` to run the pipeline.
