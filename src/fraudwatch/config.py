"""Typed configuration loaded from YAML files in configs/."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

SPLIT_ORDER = ("train", "gap", "validation", "test", "stream")


class SplitRange(BaseModel):
    """Half-open range [start_day, end_day) on the event_day axis. end_day=None means open-ended."""

    model_config = ConfigDict(frozen=True)

    start_day: float
    end_day: float | None = None

    def contains(self, day: float) -> bool:
        return day >= self.start_day and (self.end_day is None or day < self.end_day)


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_dir: Path
    processed_dir: Path
    # TransactionDT is seconds from an unknown reference; the first value in the data is 86400.
    # event_day = (TransactionDT - dt_origin_seconds) / 86400, so the data starts at day 0.
    dt_origin_seconds: int = 86_400
    splits: dict[str, SplitRange]
    # Splits computed but not written (the gap mimics label delay at training time).
    drop_splits: tuple[str, ...] = ("gap",)
    expected_fraud_rate: tuple[float, float] = (0.02, 0.06)

    @model_validator(mode="after")
    def _check_splits(self) -> DataConfig:
        if tuple(self.splits) != SPLIT_ORDER:
            raise ValueError(
                f"splits must be exactly {SPLIT_ORDER} in order, got {tuple(self.splits)}"
            )
        ranges = list(self.splits.values())
        for name, r in self.splits.items():
            if r.end_day is not None and r.end_day <= r.start_day:
                raise ValueError(f"split '{name}' is empty or inverted: {r}")
        for prev, nxt in pairwise(ranges):
            if prev.end_day is None or prev.end_day != nxt.start_day:
                raise ValueError("split ranges must be contiguous and non-overlapping")
        if ranges[-1].end_day is not None:
            raise ValueError("the last split (stream) must be open-ended (end_day: null)")
        unknown = set(self.drop_splits) - set(self.splits)
        if unknown:
            raise ValueError(f"drop_splits references unknown splits: {unknown}")
        return self

    @property
    def kept_splits(self) -> list[str]:
        return [s for s in self.splits if s not in self.drop_splits]

    def fingerprint(self) -> str:
        """Stable hash of the config contents (paths excluded), recorded in the data manifest."""
        payload = self.model_dump(mode="json", exclude={"raw_dir", "processed_dir"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_data_config(path: Path | str = "configs/data.yaml", **overrides: object) -> DataConfig:
    raw = yaml.safe_load(Path(path).read_text())
    raw.update(overrides)
    return DataConfig.model_validate(raw)
