"""Forecasting models for investment amount prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class InvestmentAmountForecaster:
    """Wrapper around a scikit-learn pipeline for investment prediction."""

    estimator: Optional[object] = None
    numeric_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"
    random_state: int = 42

    def __post_init__(self) -> None:
        base_estimator = self.estimator or RandomForestRegressor(
            n_estimators=300, random_state=self.random_state
        )

        numeric_processor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.numeric_imputer_strategy)),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_processor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.categorical_imputer_strategy)),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse=False),
                ),
            ]
        )

        preprocess = ColumnTransformer(
            transformers=[
                ("num", numeric_processor, make_column_selector(dtype_include=np.number)),
                ("cat", categorical_processor, make_column_selector(dtype_include=object)),
            ]
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", base_estimator),
            ]
        )

    def fit(self, X: pd.DataFrame, y: Sequence[float]) -> "InvestmentAmountForecaster":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def get_feature_names(self) -> Iterable[str]:
        """Return model feature names after preprocessing."""

        preprocess: ColumnTransformer = self.pipeline.named_steps["preprocess"]
        numeric_features = preprocess.transformers_[0][2]
        categorical_transformer = preprocess.transformers_[1][1]
        categorical_features = preprocess.transformers_[1][2]
        encoded_categories = categorical_transformer.named_steps["encoder"].get_feature_names_out(
            categorical_features
        )
        return list(numeric_features) + list(encoded_categories)
