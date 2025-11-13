"""Utilities for processing investment and policy datasets."""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


REQUIRED_INVESTMENT_COLUMNS: Sequence[str] = (
    "company_name",
    "company_country",
    "target_country",
    "investment_date",
    "investment_amount",
)


def prepare_investment_frame(
    raw: pd.DataFrame, *, allow_missing_amounts: bool = False
) -> pd.DataFrame:
    """Validate and normalize an investment dataframe.

    Parameters
    ----------
    raw:
        A dataframe containing investment transactions. The input is copied and
        converted to a canonical schema. Mandatory columns include the company
        identifiers, transaction date, and invested amount. Additional columns
        are preserved when present.

    Returns
    -------
    pandas.DataFrame
        A dataframe with normalized column names and an ``investment_year``
        column extracted from ``investment_date``.
    """

    missing = [col for col in REQUIRED_INVESTMENT_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"Investment dataframe is missing required columns: {missing}")

    df = raw.copy()
    df["investment_date"] = pd.to_datetime(df["investment_date"], errors="coerce")
    if df["investment_date"].isna().any():
        invalid = df[df["investment_date"].isna()]
        raise ValueError(
            "Investment dataframe contains invalid dates after coercion. "
            f"Rows: {invalid.index.tolist()}"
        )

    df["investment_year"] = df["investment_date"].dt.year.astype(int)
    # Ensure consistent casing for categorical identifiers
    df["company_name"] = df["company_name"].astype(str).str.strip()
    df["company_country"] = df["company_country"].astype(str).str.strip()
    df["target_country"] = df["target_country"].astype(str).str.strip()

    numeric_cols: Iterable[str] = ("investment_amount",)
    for col in numeric_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid_mask = converted.isna() & ~pd.isna(df[col])
        if invalid_mask.any():
            raise ValueError(
                "Investment dataframe contains non-numeric values in column "
                f"'{col}'. Rows: {df.index[invalid_mask].tolist()}"
            )

        if not allow_missing_amounts and converted.isna().any():
            raise ValueError(
                "Investment dataframe contains missing values in column "
                f"'{col}'. Set allow_missing_amounts=True to permit this."
            )

        df[col] = converted

    jobs_col = "jobs_created"
    if jobs_col in df.columns:
        df[jobs_col] = pd.to_numeric(df[jobs_col], errors="coerce")

    return df


def prepare_policy_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a policy dataframe."""

    required = {"country", "year"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Policy dataframe is missing required columns: {sorted(missing)}")

    df = raw.copy()
    df["country"] = df["country"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)

    # Ensure numerical indicators are floats
    indicator_cols = [c for c in df.columns if c not in ("country", "year")]
    df[indicator_cols] = df[indicator_cols].apply(pd.to_numeric, errors="coerce")

    return df
