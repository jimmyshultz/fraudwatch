from __future__ import annotations

from pathlib import Path

import pytest

from fraudwatch.config import DataConfig, load_data_config
from fraudwatch.data.synthetic import write_raw

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"
SYNTHETIC_ROWS = 12_000


@pytest.fixture(scope="session")
def synthetic_raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    raw_dir = tmp_path_factory.mktemp("raw")
    write_raw(raw_dir, n_rows=SYNTHETIC_ROWS, seed=0)
    return raw_dir


@pytest.fixture(scope="session")
def synthetic_config(
    synthetic_raw_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> DataConfig:
    """The real configs/data.yaml, pointed at synthetic raw data and a temp output dir."""
    return load_data_config(
        DATA_CONFIG_PATH,
        raw_dir=synthetic_raw_dir,
        processed_dir=tmp_path_factory.mktemp("processed"),
    )


@pytest.fixture(scope="session")
def real_config() -> DataConfig:
    cfg = load_data_config(DATA_CONFIG_PATH)
    raw_dir = REPO_ROOT / cfg.raw_dir
    if not (raw_dir / "train_transaction.csv").exists():
        pytest.skip("real IEEE-CIS data not present in data/raw (run `make data-download`)")
    return load_data_config(
        DATA_CONFIG_PATH, raw_dir=raw_dir, processed_dir=REPO_ROOT / cfg.processed_dir
    )
