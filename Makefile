.PHONY: install test test-all lint fmt data data-download data-build eda

install:
	uv sync --all-extras

test:
	uv run pytest -m "not slow and not realdata"

test-all:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

fmt:
	uv run ruff check --fix .
	uv run ruff format .

data: data-download data-build

data-download:
	uv run fraudwatch data download

data-build:
	uv run fraudwatch data build

eda:
	uv run --group eda jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/01_eda.ipynb
