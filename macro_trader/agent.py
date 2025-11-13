"""Company-level investment prediction agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data_processing import prepare_investment_frame, prepare_policy_frame
from .feature_engineering import FeatureBuilder
from .forecasting import InvestmentAmountForecaster


@dataclass
class CompanyInvestmentAgent:
    """Agent that predicts a company's future investments for a target country."""

    company_name: str
    home_country: Optional[str] = None
    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    forecaster: InvestmentAmountForecaster = field(
        default_factory=InvestmentAmountForecaster
    )
    default_years_ahead: Sequence[int] = (3, 4, 5)

    def __post_init__(self) -> None:
        self._fitted: bool = False
        self._investment_history: Optional[pd.DataFrame] = None
        self._policy_history: Optional[pd.DataFrame] = None
        self._training_features: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def fit(self, investments: pd.DataFrame, policy: pd.DataFrame) -> "CompanyInvestmentAgent":
        """Fit the agent to the company's historical investments."""

        company_df = investments[investments["company_name"] == self.company_name]
        if company_df.empty:
            raise ValueError(
                f"No investment records found for company '{self.company_name}'."
            )

        inferred_home = company_df["company_country"].mode().iloc[0]
        if self.home_country is None:
            self.home_country = inferred_home
        elif self.home_country != inferred_home:
            # Keep provided value but warn via exception? We'll just trust provided.
            pass

        processed_investments = prepare_investment_frame(company_df)
        processed_policy = prepare_policy_frame(policy)

        self._investment_history = processed_investments
        self._policy_history = processed_policy

        feature_frame = self.feature_builder.build_feature_frame(
            processed_investments, processed_policy
        )
        self._training_features = feature_frame

        X_train = feature_frame.drop(columns=["investment_amount", "investment_date"], errors="ignore")
        y_train = feature_frame["investment_amount"].values

        self.forecaster.fit(X_train, y_train)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(
        self,
        target_country: str,
        policy_scenario: pd.DataFrame,
        *,
        years_ahead: Optional[Sequence[int]] = None,
        base_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Predict future investment amounts for a target country."""

        if not self._fitted or self._investment_history is None or self._policy_history is None:
            raise RuntimeError("Agent must be fitted before prediction.")

        years_ahead = years_ahead or self.default_years_ahead
        if not years_ahead:
            raise ValueError("`years_ahead` must contain at least one horizon.")

        base_year = base_year or int(self._investment_history["investment_year"].max())
        scenario_policy = prepare_policy_frame(policy_scenario)
        combined_policy = pd.concat([self._policy_history, scenario_policy], ignore_index=True)
        combined_policy = combined_policy.drop_duplicates(subset=["country", "year"], keep="last")

        working_history = self._investment_history.copy()
        results: List[Dict[str, float]] = []

        for step in sorted(years_ahead):
            forecast_year = base_year + step
            placeholder_row = {
                "company_name": self.company_name,
                "company_country": self.home_country,
                "target_country": target_country,
                "investment_date": pd.Timestamp(forecast_year, 12, 31),
                "investment_amount": np.nan,
            }

            working_history = pd.concat(
                [working_history, pd.DataFrame([placeholder_row])], ignore_index=True
            )

            feature_frame = self.feature_builder.build_feature_frame(
                working_history,
                combined_policy,
                allow_missing_amounts=True,
            )
            future_features = feature_frame[
                (feature_frame["investment_year"] == forecast_year)
                & (feature_frame["target_country"] == target_country)
            ]
            if future_features.empty:
                raise RuntimeError(
                    "Unable to generate features for forecast year "
                    f"{forecast_year} and target country '{target_country}'."
                )

            X_pred = future_features.drop(
                columns=["investment_amount", "investment_date"], errors="ignore"
            )
            prediction = float(self.forecaster.predict(X_pred)[0])
            results.append(
                {
                    "company_name": self.company_name,
                    "target_country": target_country,
                    "year": forecast_year,
                    "predicted_investment_amount": prediction,
                }
            )

            # Update the placeholder with predicted value for subsequent lags
            working_history.loc[
                (working_history["investment_year"] == forecast_year)
                & (working_history["target_country"] == target_country),
                "investment_amount",
            ] = prediction

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    @property
    def training_feature_importances(self) -> Optional[pd.Series]:
        """Return feature importances when supported by the estimator."""

        if not self._fitted or self._training_features is None:
            return None

        model = self.forecaster.pipeline.named_steps["model"]
        if not hasattr(model, "feature_importances_"):
            return None

        preprocess = self.forecaster.pipeline.named_steps["preprocess"]
        numeric_features = preprocess.transformers_[0][2]
        categorical_transformer = preprocess.transformers_[1][1]
        categorical_features = preprocess.transformers_[1][2]
        encoded_categories = categorical_transformer.named_steps["encoder"].get_feature_names_out(
            categorical_features
        )
        feature_names = list(numeric_features) + list(encoded_categories)
        importances = model.feature_importances_
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)
