"""Ingest -> validate -> time-based split -> Parquet + manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from fraudwatch.config import DataConfig
from fraudwatch.data import columns as c
from fraudwatch.data.manifest import Manifest, SplitInfo
from fraudwatch.data.schema import DataValidationError, validate_raw

SECONDS_PER_DAY = 86_400


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(df: pd.DataFrame) -> str:
    """Hash of column names, dtypes, and row values; independent of Parquet writer metadata."""
    digest = hashlib.sha256()
    digest.update(repr([(col, str(dtype)) for col, dtype in df.dtypes.items()]).encode())
    digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def load_raw(raw_dir: Path, dt_origin_seconds: int = SECONDS_PER_DAY) -> pd.DataFrame:
    """Read both raw CSVs, left-join identity onto transactions, add event_day, sort by time."""
    transactions = pd.read_csv(raw_dir / c.TRANSACTION_FILE, low_memory=False)
    identity = pd.read_csv(raw_dir / c.IDENTITY_FILE, low_memory=False)
    df = transactions.merge(identity, on=c.ID_COL, how="left", validate="one_to_one")
    event_day = ((df[c.TIME_COL] - dt_origin_seconds) / SECONDS_PER_DAY).rename(c.EVENT_DAY_COL)
    df = pd.concat(
        [df, event_day], axis=1
    )  # concat, not insert: the 434-column frame is fragmented
    return df.sort_values([c.TIME_COL, c.ID_COL], kind="stable").reset_index(drop=True)


def split_by_time(df: pd.DataFrame, cfg: DataConfig) -> dict[str, pd.DataFrame]:
    """Assign every row to exactly one configured split (including dropped ones) by event_day."""
    day = df[c.EVENT_DAY_COL]
    parts: dict[str, pd.DataFrame] = {}
    assigned = pd.Series(False, index=df.index)
    for name, r in cfg.splits.items():
        mask = day >= r.start_day
        if r.end_day is not None:
            mask &= day < r.end_day
        parts[name] = df.loc[mask].reset_index(drop=True)
        assigned |= mask
    if not assigned.all():
        raise DataValidationError(
            f"{int((~assigned).sum())} rows fall outside every configured split "
            f"(event_day range {day.min():.2f}..{day.max():.2f})"
        )
    return parts


def _split_info(df: pd.DataFrame, path: str) -> SplitInfo:
    frauds = int(df[c.TARGET_COL].sum())
    return SplitInfo(
        path=path,
        rows=len(df),
        frauds=frauds,
        fraud_rate=frauds / len(df),
        min_day=float(df[c.EVENT_DAY_COL].min()),
        max_day=float(df[c.EVENT_DAY_COL].max()),
        content_sha256=content_sha256(df),
    )


def build_dataset(cfg: DataConfig) -> Manifest:
    """Full Phase 1 pipeline: load, validate, split, write kept splits as Parquet, write manifest.

    Validation runs before anything is written; the manifest is written last, so a manifest on
    disk always describes a complete, validated build.
    """
    df = validate_raw(load_raw(cfg.raw_dir, cfg.dt_origin_seconds))
    parts = split_by_time(df, cfg)

    lo, hi = cfg.expected_fraud_rate
    infos: dict[str, SplitInfo] = {}
    for name in cfg.kept_splits:
        part = parts[name]
        if part.empty:
            raise DataValidationError(f"split '{name}' is empty")
        info = _split_info(part, f"{name}.parquet")
        if not lo <= info.fraud_rate <= hi:
            raise DataValidationError(
                f"split '{name}' fraud rate {info.fraud_rate:.4f} outside expected [{lo}, {hi}]"
            )
        infos[name] = info

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    for name, info in infos.items():
        parts[name].to_parquet(cfg.processed_dir / info.path, index=False)

    manifest = Manifest(
        config_fingerprint=cfg.fingerprint(),
        raw_files={
            name: file_sha256(cfg.raw_dir / name) for name in (c.TRANSACTION_FILE, c.IDENTITY_FILE)
        },
        total_rows=len(df),
        splits=infos,
        dropped_rows={name: len(parts[name]) for name in cfg.drop_splits},
    )
    manifest.write(cfg.processed_dir)
    return manifest
