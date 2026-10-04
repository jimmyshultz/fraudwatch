"""Data manifest: what was built, from which inputs, with which config. Written next to the splits."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

MANIFEST_FILE = "manifest.json"


class SplitInfo(BaseModel):
    path: str  # relative to processed_dir
    rows: int
    frauds: int
    fraud_rate: float
    min_day: float
    max_day: float
    content_sha256: str  # hash of row contents (independent of Parquet writer metadata)


class Manifest(BaseModel):
    config_fingerprint: str
    raw_files: dict[str, str]  # file name -> sha256 of raw bytes
    total_rows: int
    splits: dict[str, SplitInfo]  # written splits only
    dropped_rows: dict[str, int]  # split name -> rows computed but not written (e.g. gap)

    def write(self, processed_dir: Path) -> Path:
        path = processed_dir / MANIFEST_FILE
        path.write_text(self.model_dump_json(indent=2) + "\n")
        return path

    @classmethod
    def read(cls, processed_dir: Path) -> Manifest:
        return cls.model_validate_json((processed_dir / MANIFEST_FILE).read_text())
