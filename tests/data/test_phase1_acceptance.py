"""Phase 1 acceptance tests (PLAN.md §11, Phase 1).

LOCKED once approved: implementation must make these pass without weakening them.
Changes to this file require a separate, explicitly reviewed commit.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fraudwatch.config import DataConfig, load_data_config
from fraudwatch.data import columns as c
from fraudwatch.data.manifest import Manifest
from fraudwatch.data.pipeline import build_dataset, load_raw, split_by_time
from fraudwatch.data.schema import DataValidationError, validate_raw
from tests.conftest import DATA_CONFIG_PATH, SYNTHETIC_ROWS

pytestmark = pytest.mark.locked

WRITTEN_SPLITS = ["train", "validation", "test", "stream"]


@pytest.fixture(scope="module")
def built(synthetic_config: DataConfig) -> tuple[DataConfig, Manifest]:
    return synthetic_config, build_dataset(synthetic_config)


def _read_split(cfg: DataConfig, manifest: Manifest, name: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.processed_dir / manifest.splits[name].path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Config -------------------------------------------------------------------------------------


def test_repo_config_has_plan_boundaries() -> None:
    cfg = load_data_config(DATA_CONFIG_PATH)
    bounds = {k: (v.start_day, v.end_day) for k, v in cfg.splits.items()}
    assert bounds == {
        "train": (0, 95),
        "gap": (95, 102),
        "validation": (102, 132),
        "test": (132, 153),
        "stream": (153, None),
    }
    assert cfg.kept_splits == WRITTEN_SPLITS


def test_config_rejects_overlapping_splits() -> None:
    with pytest.raises(ValidationError):
        load_data_config(
            DATA_CONFIG_PATH,
            splits={
                "train": {"start_day": 0, "end_day": 100},
                "gap": {"start_day": 95, "end_day": 102},
                "validation": {"start_day": 102, "end_day": 132},
                "test": {"start_day": 132, "end_day": 153},
                "stream": {"start_day": 153, "end_day": None},
            },
        )


# --- Ingest / join --------------------------------------------------------------------------------


def test_identity_is_left_joined_without_losing_or_duplicating_rows(synthetic_raw_dir: Path) -> None:
    df = load_raw(synthetic_raw_dir)
    identity = pd.read_csv(synthetic_raw_dir / c.IDENTITY_FILE)
    assert len(df) == SYNTHETIC_ROWS
    assert df[c.ID_COL].is_unique
    assert set(c.TRANSACTION_COLS) | set(c.IDENTITY_COLS) <= set(df.columns)
    assert int(df["DeviceType"].notna().sum()) == int(identity["DeviceType"].notna().sum())


def test_event_day_is_derived_from_transaction_dt(synthetic_raw_dir: Path) -> None:
    df = load_raw(synthetic_raw_dir)
    expected = (df[c.TIME_COL] - 86_400) / 86_400
    np.testing.assert_allclose(df[c.EVENT_DAY_COL].to_numpy(), expected.to_numpy())
    assert df[c.TIME_COL].is_monotonic_increasing


# --- Schema validation ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_sample(synthetic_raw_dir: Path) -> pd.DataFrame:
    return load_raw(synthetic_raw_dir).head(500).copy()


def test_valid_raw_data_passes(raw_sample: pd.DataFrame) -> None:
    validate_raw(raw_sample)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        (c.TARGET_COL, 2),
        (c.AMOUNT_COL, -5.0),
        (c.AMOUNT_COL, np.nan),
        ("ProductCD", "Z"),
        ("card6", "prepaid-unknown"),
        ("M4", "T"),
        ("DeviceType", "tablet"),
    ],
)
def test_invalid_values_are_rejected(raw_sample: pd.DataFrame, column: str, bad_value: object) -> None:
    # Inject into the column's existing dtype so only the value check can fail, not the dtype check.
    bad = raw_sample.copy()
    bad.loc[bad.index[0], column] = bad_value
    with pytest.raises(DataValidationError):
        validate_raw(bad)


def test_missing_required_column_is_rejected(raw_sample: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        validate_raw(raw_sample.drop(columns=[c.AMOUNT_COL]))


def test_duplicate_transaction_id_is_rejected(raw_sample: pd.DataFrame) -> None:
    bad = raw_sample.copy()
    bad.loc[bad.index[1], c.ID_COL] = bad.loc[bad.index[0], c.ID_COL]
    with pytest.raises(DataValidationError):
        validate_raw(bad)


# --- Splitting ------------------------------------------------------------------------------------


def test_split_assigns_every_row_to_exactly_one_split(synthetic_config: DataConfig) -> None:
    df = load_raw(synthetic_config.raw_dir)
    parts = split_by_time(df, synthetic_config)
    assert list(parts) == list(synthetic_config.splits)
    assert sum(len(p) for p in parts.values()) == len(df)
    all_ids = pd.concat([p[c.ID_COL] for p in parts.values()])
    assert all_ids.is_unique


def test_written_splits_and_gap_dropped(built: tuple[DataConfig, Manifest]) -> None:
    cfg, manifest = built
    assert list(manifest.splits) == WRITTEN_SPLITS
    for name in WRITTEN_SPLITS:
        assert (cfg.processed_dir / manifest.splits[name].path).exists()
    assert manifest.dropped_rows["gap"] > 0
    assert not any("gap" in p.name for p in cfg.processed_dir.iterdir())


def test_no_rows_lost(built: tuple[DataConfig, Manifest]) -> None:
    _, manifest = built
    written = sum(s.rows for s in manifest.splits.values())
    assert written + sum(manifest.dropped_rows.values()) == manifest.total_rows == SYNTHETIC_ROWS


def test_splits_respect_configured_day_ranges(built: tuple[DataConfig, Manifest]) -> None:
    cfg, manifest = built
    for name in WRITTEN_SPLITS:
        days = _read_split(cfg, manifest, name)[c.EVENT_DAY_COL]
        r = cfg.splits[name]
        assert days.min() >= r.start_day
        if r.end_day is not None:
            assert days.max() < r.end_day


def test_splits_are_strictly_ordered_in_time(built: tuple[DataConfig, Manifest]) -> None:
    cfg, manifest = built
    frames = [_read_split(cfg, manifest, n) for n in WRITTEN_SPLITS]
    for earlier, later in pairwise(frames):
        assert earlier[c.TIME_COL].max() < later[c.TIME_COL].min()


def test_gap_separates_train_and_validation(built: tuple[DataConfig, Manifest]) -> None:
    cfg, manifest = built
    gap = cfg.splits["gap"]
    assert gap.end_day is not None
    train_max = _read_split(cfg, manifest, "train")[c.EVENT_DAY_COL].max()
    val_min = _read_split(cfg, manifest, "validation")[c.EVENT_DAY_COL].min()
    assert train_max < gap.start_day
    assert val_min >= gap.end_day


def test_no_transaction_id_in_more_than_one_split(built: tuple[DataConfig, Manifest]) -> None:
    cfg, manifest = built
    ids = pd.concat([_read_split(cfg, manifest, n)[c.ID_COL] for n in WRITTEN_SPLITS])
    assert ids.is_unique


def test_fraud_rate_within_expected_bounds(built: tuple[DataConfig, Manifest]) -> None:
    cfg, manifest = built
    lo, hi = cfg.expected_fraud_rate
    for name, info in manifest.splits.items():
        assert lo <= info.fraud_rate <= hi, f"{name}: fraud rate {info.fraud_rate:.4f}"


# --- Manifest & reproducibility ------------------------------------------------------------------


def test_manifest_records_inputs_and_config(built: tuple[DataConfig, Manifest]) -> None:
    cfg, manifest = built
    assert manifest.config_fingerprint == cfg.fingerprint()
    assert manifest.raw_files == {
        c.TRANSACTION_FILE: _sha256(cfg.raw_dir / c.TRANSACTION_FILE),
        c.IDENTITY_FILE: _sha256(cfg.raw_dir / c.IDENTITY_FILE),
    }
    assert Manifest.read(cfg.processed_dir) == manifest


def test_manifest_stats_match_written_files(built: tuple[DataConfig, Manifest]) -> None:
    cfg, manifest = built
    for name, info in manifest.splits.items():
        df = _read_split(cfg, manifest, name)
        assert info.rows == len(df)
        assert info.frauds == int(df[c.TARGET_COL].sum())
        assert info.min_day == pytest.approx(df[c.EVENT_DAY_COL].min())
        assert info.max_day == pytest.approx(df[c.EVENT_DAY_COL].max())


def test_rebuild_is_deterministic(
    built: tuple[DataConfig, Manifest], tmp_path: Path
) -> None:
    cfg, first = built
    second = build_dataset(cfg.model_copy(update={"processed_dir": tmp_path}))
    assert second == first


def test_build_refuses_invalid_raw_data(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    tx = pd.read_csv(synthetic_raw_dir / c.TRANSACTION_FILE)
    tx.loc[0, c.TARGET_COL] = 7
    tx.to_csv(raw_dir / c.TRANSACTION_FILE, index=False)
    (raw_dir / c.IDENTITY_FILE).write_bytes((synthetic_raw_dir / c.IDENTITY_FILE).read_bytes())
    cfg = load_data_config(DATA_CONFIG_PATH, raw_dir=raw_dir, processed_dir=tmp_path / "out")
    with pytest.raises(DataValidationError):
        build_dataset(cfg)
    assert not (tmp_path / "out" / "manifest.json").exists()


# --- Real data (local only; skipped when data/raw is absent, never runs in CI) ----------------------


@pytest.mark.realdata
@pytest.mark.slow
def test_real_data_shape_and_splits(real_config: DataConfig) -> None:
    manifest = build_dataset(real_config)
    assert manifest.total_rows == 590_540
    overall = sum(s.frauds for s in manifest.splits.values()) / sum(
        s.rows for s in manifest.splits.values()
    )
    assert 0.03 <= overall <= 0.04
    assert manifest.splits["stream"].max_day > 180
    for name in WRITTEN_SPLITS:
        assert manifest.splits[name].rows > 50_000, name
