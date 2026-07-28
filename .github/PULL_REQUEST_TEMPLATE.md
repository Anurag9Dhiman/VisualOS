## Summary
<!-- 1–3 bullets on what changed and why -->
-

## Type
<!-- Mark all that apply -->
- [ ] Feature
- [ ] Bug fix
- [ ] Refactor / cleanup
- [ ] Test coverage
- [ ] Docs / config

## Test plan
<!-- How did you verify this? Check everything you actually ran -->
- [ ] `make test` passes (133+ tests, ≥90% coverage)
- [ ] `make lint` clean
- [ ] `make typecheck` clean
- [ ] `make security` clean
- [ ] Manually tested with `make run IMAGE=…` or `make inspect`
- [ ] Integration tests pass (`make integration IMAGE=…`)

## Hard rules checklist
<!-- Every PR must confirm these — see CLAUDE.md -->
- [ ] No API keys committed
- [ ] `src/prompts.py` not modified (or explicit human review done)
- [ ] No agent's responsibilities expanded
- [ ] `failure_modes_checked` field present on any new VisionResult usage
- [ ] Every new external call wrapped in `asyncio.wait_for(...)`
- [ ] No user frames stored beyond 30 days
