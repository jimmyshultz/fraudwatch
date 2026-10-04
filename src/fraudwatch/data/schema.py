"""Pandera schema for the joined raw IEEE-CIS data (transactions left-joined with identity)."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from fraudwatch.data import columns as c


class DataValidationError(ValueError):
    """Schema validation failure. Wraps pandera errors so callers needn't depend on pandera."""


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


_numeric = pa.Check(_is_numeric, element_wise=False, error="column must be numeric")


def _category(values: list[str]) -> pa.Column:
    return pa.Column(checks=pa.Check.isin(values), nullable=True)


def _numeric_col(nullable: bool = True, *checks: pa.Check) -> pa.Column:
    return pa.Column(checks=[_numeric, *checks], nullable=nullable)


def _build_schema() -> pa.DataFrameSchema:
    cols: dict[str, pa.Column] = {
        c.ID_COL: pa.Column(int, unique=True, nullable=False),
        c.TARGET_COL: pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False),
        c.TIME_COL: pa.Column(int, checks=pa.Check.ge(0), nullable=False),
        c.AMOUNT_COL: _numeric_col(False, pa.Check.ge(0)),
        "ProductCD": pa.Column(checks=pa.Check.isin(c.PRODUCT_CODES), nullable=False),
        "card4": _category(c.CARD4_VALUES),
        "card6": _category(c.CARD6_VALUES),
        "P_emaildomain": pa.Column(nullable=True),
        "R_emaildomain": pa.Column(nullable=True),
        "DeviceType": _category(c.DEVICE_TYPES),
        "DeviceInfo": pa.Column(nullable=True),
    }
    numeric = [
        "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2",
        *c.C_COLS, *c.D_COLS, *c.V_COLS, *c.NUMERIC_ID_COLS,
    ]  # fmt: skip
    cols.update({name: _numeric_col() for name in numeric})
    cols.update({m: _category(c.M4_VALUES if m == "M4" else c.M_FLAG_VALUES) for m in c.M_COLS})
    cols.update({name: pa.Column(nullable=True) for name in c.CATEGORICAL_ID_COLS})
    # strict=False: derived columns (e.g. event_day) are allowed; every raw column is required.
    return pa.DataFrameSchema(cols, strict=False, name="ieee_cis_raw")


RAW_SCHEMA = _build_schema()


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the joined transaction+identity frame; raise DataValidationError on failure."""
    try:
        return RAW_SCHEMA.validate(df, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases
        summary = cases.groupby(["column", "check"], dropna=False).size().head(20).to_string()
        raise DataValidationError(f"raw data failed validation:\n{summary}") from exc
    except SchemaError as exc:
        raise DataValidationError(f"raw data failed validation: {exc}") from exc
