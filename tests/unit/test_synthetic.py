from __future__ import annotations

import pandas as pd

from fraudwatch.data import columns as c
from fraudwatch.data.synthetic import DT_END, DT_START, generate_raw


def test_columns_match_raw_layout() -> None:
    tx, ident = generate_raw(n_rows=500, seed=1)
    assert list(tx.columns) == c.TRANSACTION_COLS
    assert list(ident.columns) == c.IDENTITY_COLS
    assert len(c.TRANSACTION_COLS) == 394
    assert len(c.IDENTITY_COLS) == 41


def test_same_seed_is_deterministic() -> None:
    a_tx, a_id = generate_raw(n_rows=500, seed=7)
    b_tx, b_id = generate_raw(n_rows=500, seed=7)
    pd.testing.assert_frame_equal(a_tx, b_tx)
    pd.testing.assert_frame_equal(a_id, b_id)


def test_shape_of_generated_data() -> None:
    tx, ident = generate_raw(n_rows=20_000, seed=0)
    assert 0.03 <= tx[c.TARGET_COL].mean() <= 0.04
    assert tx[c.TIME_COL].between(DT_START, DT_END).all()
    assert tx[c.TIME_COL].is_monotonic_increasing
    assert 0.2 <= len(ident) / len(tx) <= 0.35
    assert set(ident[c.ID_COL]) <= set(tx[c.ID_COL])


def test_planted_signal_exists() -> None:
    tx, _ = generate_raw(n_rows=20_000, seed=0)
    fraud = tx[c.TARGET_COL] == 1
    assert tx.loc[fraud, c.AMOUNT_COL].median() > 1.5 * tx.loc[~fraud, c.AMOUNT_COL].median()
    assert (tx.loc[fraud, "ProductCD"] == "C").mean() > 2 * (
        tx.loc[~fraud, "ProductCD"] == "C"
    ).mean()
