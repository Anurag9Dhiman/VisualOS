# Lens OS

Point your camera at any building, monument, or object → get its identification, history, live info, and personalised facts in under 3 seconds.

**Phase 0** — full pipeline running as a Python script + Streamlit inspector. The goal is to validate the loop before writing production iOS code.

---

## How it works

```
Image + GPS
     │
     ▼
┌─────────────────────────────────────────────────┐
│ Orchestrator (LangGraph)                        │
│                                                 │
│  plan → cache_check → vision_memory → fuse → write_memory
│               │              │                  │
│           (HIT) skip     confidence_gate        ▼
│               │            │      │         (async)
│               │         search  (skip)
│               └────────────▼────┘
│                           fuse
└─────────────────────────────────────────────────┘
     │
     ▼
ResponseCard  (NormalCard | FallbackCard)
```

Three specialist agents run in parallel:

| Agent | Model | Job |
|---|---|---|
| Vision | `gemini-2.0-flash` | Identifies the entity from the image using 5-step chain-of-thought reasoning |
| Memory | `text-embedding-004` | Retrieves past interactions and user interests from SQLite |
| Search | `gemini-2.0-flash` | ReAct loop (≤3 tool calls) — Wikipedia, Wikidata, Tavily, OSM |

Fusion composes a single card from all three outputs. Memory writes happen after the user already has their answer.

**Latency budget:** Vision (800ms) + Memory (400ms) run in parallel. Search (800ms) only runs if Vision is confident. Overall hard timeout: 2.5s.

---

## Setup

```bash
# 1. Clone and create virtualenv
git clone https://github.com/Anurag9Dhiman/VisualOS.git
cd VisualOS
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API keys
cp .env.example .env
# Edit .env — GOOGLE_API_KEY is required; TAVILY_API_KEY is optional
```

**Required:** `GOOGLE_API_KEY` — used by all agents (Gemini 2.0 Flash + text-embedding-004).  
**Optional:** `TAVILY_API_KEY` — enables live web search in the Search agent (falls back gracefully without it).

---

## Usage

### CLI (streaming by default)

```bash
python -m src.main --image path/to/photo.jpg --lat 12.95 --lng 77.58
```

Flags:
- `--image` — path to the photo (JPEG, PNG, etc.)
- `--lat`, `--lng` — GPS coordinates (optional; improves identification accuracy)
- `--user-id` — user identifier for memory/personalisation (default: `default`)
- `--json` — output raw JSON instead of streamed text

### Web inspector

```bash
streamlit run src/inspector.py
```

Upload an image, set coordinates, and run the pipeline. Results shown across five tabs: Vision, Memory, Search, Cost, Raw JSON.

---

## Tests

All tests run offline — no API keys needed.

```bash
pytest                   # run all tests
pytest -v                # verbose output
pytest tests/test_agents.py  # specific file
```

Current coverage: **43+ tests** across contracts, cost logger, DB, rate limiter, agents, tools, cache, and fusion.

---

## Project structure

```
src/
  main.py           CLI entry point
  orchestrator.py   LangGraph graph — run_pipeline / stream_pipeline
  contracts.py      All Pydantic v2 models
  prompts.py        System prompts (immutable — see CLAUDE.md)
  fusion.py         Blocking and streaming fusion
  db.py             SQLite — interactions, cost_log, user_interests
  cache.py          Response cache (24h TTL, SHA256 key)
  cost_logger.py    Mandatory cost logging for every LLM call
  rate_limiter.py   Sliding-window rate limiter (15 RPM free tier)
  inspector.py      Streamlit dev inspector
  agents/
    vision.py       Vision agent + image preprocessing
    memory.py       Embedding search + interest snapshot
    search.py       ReAct search loop
  tools/
    wikipedia_client.py
    wikidata_client.py
    tavily_client.py
    osm_client.py
tests/
  conftest.py             Fresh SQLite + rate limiter reset per test
  test_contracts.py
  test_cost_logger.py
  test_db.py
  test_rate_limiter.py
  test_agents.py
  test_tools.py
  test_cache.py
  test_fusion.py
docs/
  architecture.md         Full design doc — read before making non-trivial changes
```

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full design: LangGraph node table, agent specs, confidence gate logic, Pydantic contracts, SQLite schema, cost logging format, latency budget, and file map.

---

## Hard rules (from CLAUDE.md)

- Never commit API keys.
- Never modify `src/prompts.py` without explicit human review.
- Never expand an agent's responsibilities — Vision identifies, Search researches, Memory recalls.
- Never skip `failure_modes_checked` in VisionResult.
- Never call an external service without a timeout.
- Never store user frames longer than 30 days (privacy).
