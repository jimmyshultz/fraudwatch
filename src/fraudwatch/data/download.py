"""Download the IEEE-CIS competition files from Kaggle.

Requires: the competition rules accepted on kaggle.com, and Kaggle credentials configured
(`kaggle auth login`, KAGGLE_API_TOKEN, or ~/.kaggle/access_token). Only the two training files
are extracted; the data is never committed or redistributed (see docs/adr/0001).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from fraudwatch.data import columns as c

COMPETITION = "ieee-fraud-detection"
WANTED = (c.TRANSACTION_FILE, c.IDENTITY_FILE)


class KaggleSetupError(RuntimeError):
    pass


def download_raw(raw_dir: Path, force: bool = False) -> list[Path]:
    targets = [raw_dir / name for name in WANTED]
    if not force and all(t.exists() for t in targets):
        return targets

    # Imported lazily: importing `kaggle` attempts authentication as a side effect.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit as exc:  # the client calls exit(1) when no credentials are found
        raise KaggleSetupError(
            "Kaggle credentials not found. Run `uv run kaggle auth login` or set KAGGLE_API_TOKEN."
        ) from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        api.competition_download_files(COMPETITION, path=str(raw_dir), force=force, quiet=False)
    except Exception as exc:
        raise KaggleSetupError(
            f"Download failed ({exc}). If this is a 403, accept the rules at "
            f"https://www.kaggle.com/competitions/{COMPETITION}/rules first."
        ) from exc

    archive = raw_dir / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive) as zf:
        for name in WANTED:
            zf.extract(name, raw_dir)
    archive.unlink()
    return targets
