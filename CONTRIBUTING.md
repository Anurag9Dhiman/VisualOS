# Contributing to Lens OS

Phase 0 is a Python proof-of-concept. The goal is to validate the full
identify → research → personalise → stream loop before writing any iOS code.

---

## 1. First-time setup

```bash
git clone https://github.com/Anurag9Dhiman/VisualOS.git
cd VisualOS

make setup          # creates .venv, installs deps, installs pre-commit hooks
cp .env.example .env
# fill in GOOGLE_API_KEY — required to run the pipeline
```

`make setup` installs pre-commit hooks that run `ruff --fix` and
`ruff-format` on every commit, so formatting is never a manual step.

---

## 2. Daily commands

| Command | What it does |
|---|---|
| `make test` | Run all 130+ unit tests (no API keys needed) |
| `make cov` | Tests + line-level coverage report |
| `make lint` | `ruff format --check` + `ruff check` |
| `make format` | Auto-fix formatting and import order |
| `make typecheck` | `mypy src/` |
| `make security` | `pip-audit` — check deps for known CVEs |
| `make run IMAGE=photo.jpg LAT=12.95 LNG=77.58` | Run the CLI pipeline |
| `make inspect` | Open the Streamlit web inspector |
| `make integration IMAGE=photo.jpg` | End-to-end test against real Gemini API |

---

## 3. How the pipeline works

```
Image + GPS
    │
    ▼
plan → cache_check → vision_memory ──► search ──► fuse → write_memory
                              │                    ▲
                              └── (confidence gate)┘
                              (skip search if guessing)
```

Three specialist agents run in parallel inside `vision_memory`:

- **Vision** (`src/agents/vision.py`) — identifies the entity using Gemini
  with a 5-step reasoning prompt. Never searches the web.
- **Memory** (`src/agents/memory.py`) — cosine-searches past interactions
  in SQLite. Never identifies. Never searches.
- **Search** (`src/agents/search.py`) — ReAct loop with a 3-call budget.
  Only runs if Vision confidence ≥ `uncertain`.

Fusion (`src/fusion.py`) composes all three outputs into a card and streams
it token by token.

Read `docs/architecture.md` for full detail before touching `src/orchestrator.py`.

---

## 4. Adding a feature

```bash
git checkout main && git pull
git checkout -b feat/your-feature-name
# implement
make test && make lint && make typecheck && make security
git push -u origin feat/your-feature-name
# open a PR — GitHub pre-fills the template
```

One feature per PR. PRs that mix a new feature with unrelated cleanup are
harder to review and harder to revert.

---

## 5. Hard rules (from `CLAUDE.md`)

These are non-negotiable. PRs that break any of them will not be merged.

| Rule | Why |
|---|---|
| Never commit API keys | Obvious. Use `.env` (gitignored). |
| Never modify `src/prompts.py` without human review | Changes break prompt caching and invalidate eval baselines. Propose the diff in chat first. |
| Never expand an agent's responsibilities | Vision identifies. Search researches. Memory recalls. Adding cross-cutting concerns creates hidden coupling that's hard to test and hard to remove. |
| Never skip `failure_modes_checked` in VisionResult | The field forces explicit reflection on what could be wrong. It's a required Pydantic field — the pipeline will error if absent. |
| Never call an external service without `asyncio.wait_for(...)` | The overall pipeline has a 2.5s hard deadline. An uncapped call can silently blow the budget. |
| Never store user frames longer than 30 days | Privacy requirement. `MEMORY_TTL_DAYS = 30` in `src/db.py` enforces this. Don't work around it. |

---

## 6. Writing tests

All tests live in `tests/`. No API keys are needed for the unit test suite —
the Gemini client is mocked in `tests/conftest.py`.

```bash
# run a single test file
pytest tests/test_agents.py -v

# run tests matching a name pattern
pytest -k "test_vision" -v

# run with coverage for a specific module
pytest --cov=src.agents.vision --cov-report=term-missing tests/test_agents.py
```

Integration tests (real API calls) are in `tests/test_integration.py` and
are skipped unless `GOOGLE_API_KEY` and `TEST_IMAGE_PATH` are set:

```bash
make integration IMAGE=path/to/photo.jpg LAT=12.95 LNG=77.58
```

Coverage gate is 90%. Check with `make cov` before pushing.

---

## 7. CI jobs (all blocking)

| Job | Trigger | What fails it |
|---|---|---|
| Tests | push / PR | any test failure, coverage < 90% |
| Lint | push / PR | ruff format drift, ruff rule violation |
| Type check | push / PR | any mypy error, stale `# type: ignore` |
| Security | push / PR | any CVE in installed packages |

Dependabot opens weekly PRs to update pip packages and GitHub Actions
versions. Merge these promptly to keep the security job green.
