"""fraudwatch command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from fraudwatch.config import load_data_config

app = typer.Typer(no_args_is_help=True, help="Fraud detection with drift monitoring.")
data_app = typer.Typer(no_args_is_help=True, help="Download, validate, and split the dataset.")
app.add_typer(data_app, name="data")

ConfigOpt = Annotated[Path, typer.Option("--config", help="Data config YAML.")]


@data_app.command("download")
def data_download(
    config: ConfigOpt = Path("configs/data.yaml"),
    force: Annotated[bool, typer.Option(help="Re-download even if files exist.")] = False,
) -> None:
    """Download the IEEE-CIS training files from Kaggle into raw_dir."""
    from fraudwatch.data.download import KaggleSetupError, download_raw

    cfg = load_data_config(config)
    try:
        paths = download_raw(cfg.raw_dir, force=force)
    except KaggleSetupError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    for p in paths:
        typer.echo(f"ready: {p} ({p.stat().st_size / 1e6:.0f} MB)")


@data_app.command("build")
def data_build(config: ConfigOpt = Path("configs/data.yaml")) -> None:
    """Validate raw data, split by time, write Parquet splits + manifest to processed_dir."""
    from fraudwatch.data.pipeline import build_dataset
    from fraudwatch.data.schema import DataValidationError

    cfg = load_data_config(config)
    try:
        manifest = build_dataset(cfg)
    except DataValidationError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{manifest.total_rows:,} rows -> {cfg.processed_dir}")
    for name, s in manifest.splits.items():
        typer.echo(
            f"  {name:<11} {s.rows:>8,} rows  fraud {s.fraud_rate:6.2%}  "
            f"days {s.min_day:6.1f}-{s.max_day:6.1f}"
        )
    for name, n in manifest.dropped_rows.items():
        typer.echo(f"  {name:<11} {n:>8,} rows  (dropped)")
