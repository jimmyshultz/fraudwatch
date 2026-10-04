"""Synthetic IEEE-CIS-shaped data for tests and CI.

The real dataset cannot be committed or downloaded in public CI (Kaggle competition rules),
so CI runs the full pipeline on this generator instead. It reproduces the raw file layout
(same columns, dtypes, value domains, missingness style, ~3.5% fraud, ~25% identity coverage,
~182 days of TransactionDT) and plants a learnable signal so later phases can assert that a
model beats the prior. It does NOT try to reproduce the real data's statistics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraudwatch.data import columns as c

DT_START = 86_400
DT_END = 15_811_131  # last TransactionDT in the real training file (~182 days later)
EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com",
    "comcast.net", "icloud.com", "outlook.com", "protonmail.com", "mail.com",
]  # fmt: skip
DEVICE_INFOS = ["Windows", "iOS Device", "MacOS", "Trident/7.0", "SM-G960U", "Moto G (5)"]


def _with_nans(rng: np.random.Generator, values: np.ndarray, nan_rate: float) -> np.ndarray:
    out = values.astype(float)
    out[rng.random(len(out)) < nan_rate] = np.nan
    return out


def _cat_with_nans(
    rng: np.random.Generator, choices: list[str], n: int, nan_rate: float, p: np.ndarray | None = None
) -> np.ndarray:
    out = rng.choice(np.array(choices, dtype=object), size=n, p=p)
    out[rng.random(n) < nan_rate] = None
    return out


def generate_raw(n_rows: int = 20_000, seed: int = 0, fraud_rate: float = 0.035) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transactions, identity) DataFrames with the raw Kaggle column layout."""
    rng = np.random.default_rng(seed)
    n = n_rows
    y = (rng.random(n) < fraud_rate).astype(np.int64)
    fraud = y == 1

    dt = np.sort(rng.integers(DT_START, DT_END + 1, size=n))
    tx: dict[str, object] = {
        c.ID_COL: np.arange(2_987_000, 2_987_000 + n),
        c.TARGET_COL: y,
        c.TIME_COL: dt,
    }

    # Planted signal: fraud skews to higher amounts, product C, credit cards, some email domains.
    log_amt = rng.normal(4.0, 1.0, n) + 0.8 * fraud
    tx[c.AMOUNT_COL] = np.round(np.exp(log_amt), 3)
    p_legit = np.array([0.74, 0.06, 0.10, 0.02, 0.08])
    p_fraud = np.array([0.45, 0.12, 0.33, 0.04, 0.06])
    product = np.where(
        fraud,
        rng.choice(c.PRODUCT_CODES, n, p=p_fraud),
        rng.choice(c.PRODUCT_CODES, n, p=p_legit),
    )
    tx["ProductCD"] = product.astype(object)

    tx["card1"] = rng.integers(1000, 18_397, n)
    tx["card2"] = _with_nans(rng, rng.integers(100, 601, n), 0.015)
    tx["card3"] = _with_nans(rng, rng.choice([150, 185, 106, 144], n), 0.003)
    tx["card4"] = _cat_with_nans(rng, c.CARD4_VALUES, n, 0.003, p=np.array([0.65, 0.32, 0.015, 0.015]))
    card6_legit = rng.choice(c.CARD6_VALUES, n, p=np.array([0.77, 0.229, 0.0005, 0.0005]))
    card6_fraud = rng.choice(c.CARD6_VALUES, n, p=np.array([0.50, 0.499, 0.0005, 0.0005]))
    card6 = np.where(fraud, card6_fraud, card6_legit).astype(object)
    card6[rng.random(n) < 0.003] = None
    tx["card5"] = _with_nans(rng, rng.choice([226, 224, 166, 102, 117], n), 0.007)
    tx["card6"] = card6
    tx["addr1"] = _with_nans(rng, rng.integers(100, 541, n), 0.11)
    tx["addr2"] = _with_nans(rng, rng.choice([87, 60, 96], n, p=np.array([0.98, 0.01, 0.01])), 0.11)
    tx["dist1"] = _with_nans(rng, rng.exponential(100, n).round(), 0.6)
    tx["dist2"] = _with_nans(rng, rng.exponential(200, n).round(), 0.93)
    tx["P_emaildomain"] = _cat_with_nans(rng, EMAIL_DOMAINS, n, 0.16)
    tx["R_emaildomain"] = _cat_with_nans(rng, EMAIL_DOMAINS, n, 0.77)

    for col in c.C_COLS:
        tx[col] = rng.poisson(1.0 + 1.5 * fraud).astype(float)
    for col in c.D_COLS:
        tx[col] = _with_nans(rng, rng.integers(0, 640, n), float(rng.uniform(0.1, 0.9)))
    for col in c.M_COLS:
        values = c.M4_VALUES if col == "M4" else c.M_FLAG_VALUES
        tx[col] = _cat_with_nans(rng, values, n, float(rng.uniform(0.3, 0.6)))
    v_block = rng.normal(0.0, 1.0, (n, len(c.V_COLS))).round(3)
    v_block[:, :10] += 0.7 * fraud[:, None]  # a few informative V columns
    v_block[rng.random(v_block.shape) < 0.4] = np.nan
    tx.update({col: v_block[:, i] for i, col in enumerate(c.V_COLS)})

    transactions = pd.DataFrame(tx, columns=c.TRANSACTION_COLS)

    # Identity rows for ~25% of transactions (fraud more likely to have one, as in the real data).
    has_id = rng.random(n) < np.where(fraud, 0.6, 0.24)
    m = int(has_id.sum())
    ident: dict[str, object] = {c.ID_COL: transactions.loc[has_id, c.ID_COL].to_numpy()}
    for col in c.NUMERIC_ID_COLS:
        ident[col] = _with_nans(rng, rng.normal(0, 50, m).round(), float(rng.uniform(0.0, 0.9)))
    for col in c.CATEGORICAL_ID_COLS:
        ident[col] = _cat_with_nans(rng, ["Found", "NotFound", "New"], m, float(rng.uniform(0.0, 0.9)))
    ident["DeviceType"] = _cat_with_nans(rng, c.DEVICE_TYPES, m, 0.02)
    ident["DeviceInfo"] = _cat_with_nans(rng, DEVICE_INFOS, m, 0.18)
    identity = pd.DataFrame(ident, columns=c.IDENTITY_COLS)
    return transactions, identity


def write_raw(out_dir: Path, n_rows: int = 20_000, seed: int = 0) -> tuple[Path, Path]:
    """Write synthetic raw CSVs with the Kaggle file names into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    transactions, identity = generate_raw(n_rows=n_rows, seed=seed)
    tx_path = out_dir / c.TRANSACTION_FILE
    id_path = out_dir / c.IDENTITY_FILE
    transactions.to_csv(tx_path, index=False)
    identity.to_csv(id_path, index=False)
    return tx_path, id_path
