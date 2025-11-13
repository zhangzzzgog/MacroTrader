"""Feature engineering utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .data_processing import prepare_investment_frame, prepare_policy_frame


@dataclass
class FeatureBuilder:
    """Create model-ready features from investment and policy datasets."""

    lag_years: Sequence[int] = (1, 2, 3)
    rolling_window: int = 3
    include_jobs_features: bool = True
    policy_prefix_home: str = "home_policy_"
    policy_prefix_target: str = "target_policy_"
    extra_groupby_features: Iterable[str] = field(
        default_factory=lambda: ("years_since_first", "years_since_last")
    )

    def build_feature_frame(
        self,
        investments: pd.DataFrame,
        policy: pd.DataFrame,
        *,
        allow_missing_amounts: bool = False,
    ) -> pd.DataFrame:
        """Return a dataframe ready for model consumption."""

        investment_df = prepare_investment_frame(
            investments, allow_missing_amounts=allow_missing_amounts
        )
        policy_df = prepare_policy_frame(policy)

        merged = self._merge_policy_information(investment_df, policy_df)
        engineered = self._generate_temporal_features(merged)
        return engineered

    # ------------------------------------------------------------------
    def _merge_policy_information(
        self, investments: pd.DataFrame, policy: pd.DataFrame
    ) -> pd.DataFrame:
        df = investments.copy()
        df = self._merge_policy(df, policy, "company_country", self.policy_prefix_home)
        df = self._merge_policy(df, policy, "target_country", self.policy_prefix_target)
        return df

    def _merge_policy(
        self, investments: pd.DataFrame, policy: pd.DataFrame, country_col: str, prefix: str
    ) -> pd.DataFrame:
        indicator_cols = [c for c in policy.columns if c not in {"country", "year"}]
        rename_map = {col: f"{prefix}{col}" for col in indicator_cols}
        renamed = policy.rename(
            columns={"country": country_col, "year": "investment_year", **rename_map}
        )
        merged = investments.merge(
            renamed,
            how="left",
            on=[country_col, "investment_year"],
            suffixes=("", "_y"),
        )
        duplicate_cols = [c for c in merged.columns if c.endswith("_y")]
        if duplicate_cols:
            merged = merged.drop(columns=duplicate_cols)
        return merged

    # ------------------------------------------------------------------
    def _generate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["company_name", "target_country", "investment_year"])
        group_cols = ["company_name", "target_country"]
        amount_group = df.groupby(group_cols)["investment_amount"]

        for lag in self.lag_years:
            df[f"amount_lag_{lag}"] = amount_group.shift(lag)

        shifted_amount = amount_group.shift(1)
        df["amount_rolling_mean"] = shifted_amount.rolling(
            window=self.rolling_window, min_periods=1
        ).mean()
        df["amount_rolling_std"] = shifted_amount.rolling(
            window=self.rolling_window, min_periods=1
        ).std()

        df["investment_trend"] = amount_group.shift(1)
        df["investment_growth"] = amount_group.pct_change().replace([np.inf, -np.inf], np.nan)

        df["years_since_last"] = df.groupby(group_cols)["investment_year"].diff()
        first_year = df.groupby(group_cols)["investment_year"].transform("min")
        df["years_since_first"] = df["investment_year"] - first_year

        if self.include_jobs_features and "jobs_created" in df.columns:
            jobs_group = df.groupby(group_cols)["jobs_created"]
            df["jobs_lag_1"] = jobs_group.shift(1)
            df["jobs_rolling_mean"] = jobs_group.shift(1).rolling(
                window=self.rolling_window, min_periods=1
            ).mean()

        # Past allocation share per country within the company portfolio
        df["prev_investment_amount"] = amount_group.shift(1)
        prev_total = df.groupby(["company_name", "investment_year"])["prev_investment_amount"].transform(
            "sum"
        )
        df["prev_target_share"] = df["prev_investment_amount"] / prev_total

        df = df.drop(columns=["prev_investment_amount"])

        return df
