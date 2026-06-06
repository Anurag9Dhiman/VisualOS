# Lens OS
A visual intelligence iOS app and the first product in a larger AI OS vision. Point the camera at any building, monument, statue, or object → get its identification, history, live info, and personalised facts tailored to the user — in under 3 seconds, with memory of every interaction.
**Current phase: Phase 0 — building the core loop as a Python script + web view proof of concept before writing production code.**
Read the full architecture before making non-trivial changes:
@docs/architecture.md
The three agent prompts are extracted into a clean importable module:
@src/prompts.py
---
## Architecture in one paragraph
Hierarchical Planner-Executor with parallel specialist agents. A LangGraph orchestrator plans and dispatches three specialists — Vision, Memory, Search — that run concurrently. Vision uses GPT-4o with a single-pass 5-step reasoning prompt (identify → cross-check → alternatives → failure modes → commit confidence). Search uses ReAct (reason → tool → observe, hard 3-call budget). Memory does straight vector retrieval. The orchestrator fuses outputs into one streamed response card. Memory writes happen asynchronously, *after* the user already has their answer.
Each agent has a hard timeout (800ms per specialist, 2.5s overall) to guarantee the <3s P50 latency target.
---
## Tech stack
| Layer | Choice |
|---|---|
| Backend language | Python 3.12 |
| Agent orchestration | LangGraph |
| LLM clients | `anthropic` (Claude Sonnet for search/fusion), `openai` (GPT-4o Vision, text-embedding-3-small) |
| Validation | Pydantic v2 — every agent I/O is a Pydantic model |
| Database | SQLite (Phase 0), PostgreSQL + pgvector (Phase 1) |
| Tracing | LangSmith |
| Testing | pytest + pytest-asyncio |
---
## Conventions
- **Pydantic v2** for every agent input/output — schemas in `src/contracts.py`. Never pass raw dicts between agents.
- **Async by default** — every I/O call is `async`. Agents run in parallel via `asyncio.gather`.
- **Hard timeouts at every external boundary** — wrap every LLM/HTTP call in `asyncio.wait_for(...)`.
- **Cost logging is mandatory.** Every LLM call logs `{model, input_tokens, output_tokens, cost_usd, agent}`.
- **Prompts are immutable strings in `src/prompts.py`.** Keep system messages stable for prompt caching.
---
## Hard rules
- **Never commit API keys.**
- **Never modify `src/prompts.py` without explicit human review.** Propose diffs in chat first.
- **Never expand an agent's responsibilities.** Vision identifies. Search researches. Memory recalls.
- **Never skip `failure_modes_checked`** in VisionResult — it's a required field.
- **Never call an external service without a timeout.**
- **Never store user frames longer than 30 days** (privacy requirement).
---
## Useful commands
```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in keys

# Run CLI
python -m src.main --image path/to/photo.jpg --lat 12.95 --lng 77.58

# Web inspector
streamlit run src/inspector.py

# Tests (no API keys needed)
pytest

# Type check
mypy src/
```
