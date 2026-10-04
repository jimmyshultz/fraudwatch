# ADR-0001: Use IEEE-CIS Fraud Detection as the dataset

**Status:** Accepted · 2026-10-04

## Context
The project needs real-time scoring, drift monitoring, and retraining on data with a time axis long
enough for honest time-based validation *and* a held-out month to replay as a stream.
Candidates: IEEE-CIS (Vesta, ~590k rows, ~3.5% fraud, ~6 months), PaySim (synthetic, 30 days,
balance columns nearly leak the label), ULB credit card (2 days, PCA-anonymized features).

## Decision
Use IEEE-CIS. Split by `TransactionDT` into train / 7-day gap / validation / test / stream
(boundaries in `configs/data.yaml`).

## Consequences
- Realistic, wide, partially missing data; natural drift exists (e.g. raw `D*` columns).
- Kaggle competition rules: data is downloaded by each user after accepting the rules, never
  committed, never baked into images or the Lambda package. CI runs on `fraudwatch.data.synthetic`.
- ~430 columns: a curated ~50-field input contract is needed for serving (Phase 3).
- Fallback if the rules turn out to forbid portfolio use: PaySim with balance columns removed;
  the pipeline is dataset-agnostic behind `data/columns.py` + `configs/data.yaml`.
