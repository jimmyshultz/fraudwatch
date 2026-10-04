# fraudwatch — agent guide

Fraud detection + drift monitoring portfolio project. **PLAN.md is the source of truth** for scope,
design decisions, and phase acceptance criteria. ADRs in `docs/adr/` explain the *why*; don't
re-litigate a decision that has an ADR without flagging it to the user first.

## Commands
- `make install` — `uv sync --all-extras`
- `make test` — fast tests (excludes `slow`, `realdata`)
- `make lint` — ruff check + format check + mypy
- `make data` — download (needs Kaggle token) + validate + split real data
- `uv run fraudwatch --help` — CLI

## Pinned stack (check before using an API from memory)
Python 3.14 · pandas **3.0** (copy-on-write always on; default string dtype is `str`, not `object`) ·
pandera **0.33** (`import pandera.pandas as pa`, not `import pandera as pa`) · pydantic 2 ·
numpy 2. Versions are locked in `uv.lock`; never upgrade a pinned library as a side effect of a task.

## Hard rules
1. **Never use random splits.** All splits are time-based from `configs/data.yaml`.
2. **Never commit or package data.** `data/` is gitignored; IEEE-CIS rules forbid redistribution.
   CI uses `fraudwatch.data.synthetic` only.
3. **Locked tests** (`@pytest.mark.locked`) encode approved acceptance criteria. Do not weaken,
   skip, xfail, or delete them to get green. If one looks wrong, stop and ask the user.
4. **No typed-in metrics.** Numbers in README, reports, or docs must be generated from run outputs.
5. **Leakage review required** for anything that fits on data (encoders, calibrators, thresholds):
   fit on the split PLAN.md §5 says, and add a test proving it.
6. **$0 budget.** Terraform may only declare resource types in `infra/allowed_resources.txt`.
7. Config values go in `configs/*.yaml`, loaded via `fraudwatch.config`; no magic numbers in code.

## Layout
`src/fraudwatch/` package (src layout) · `tests/{unit,data,...}` · `configs/` YAML ·
`docs/adr/` decisions · `docs/progress.md` per-phase notes written by the user.
