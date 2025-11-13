"""Utilities to create policy scenarios for forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from .data_processing import prepare_policy_frame


@dataclass
class PolicyScenarioBuilder:
    """Helper class to create future policy scenarios."""

    base_policy: pd.DataFrame

    def __post_init__(self) -> None:
        self._scenario = prepare_policy_frame(self.base_policy)

    def set_indicator_values(
        self, country: str, year: int, indicators: Mapping[str, float]
    ) -> "PolicyScenarioBuilder":
        """Set or override policy indicators for a country and year."""

        mask = (self._scenario["country"] == country) & (self._scenario["year"] == year)
        if mask.any():
            for key, value in indicators.items():
                self._scenario.loc[mask, key] = value
        else:
            base_row = {"country": country, "year": year, **indicators}
            self._scenario = pd.concat([self._scenario, pd.DataFrame([base_row])], ignore_index=True)
        return self

    def apply_growth_projection(
        self,
        country: str,
        indicator: str,
        *,
        annual_growth: float,
        years: Sequence[int],
    ) -> "PolicyScenarioBuilder":
        """Project an indicator using a compound growth rate."""

        country_data = self._scenario[self._scenario["country"] == country]
        if country_data.empty:
            raise ValueError(f"No base data available for country '{country}'.")

        last_record = country_data.sort_values("year").iloc[-1]
        last_year = int(last_record["year"])
        base_value = float(last_record.get(indicator, 0.0))

        for step, year in enumerate(sorted(years), start=1):
            if year <= last_year:
                continue
            projected = base_value * ((1.0 + annual_growth) ** step)
            self.set_indicator_values(country, year, {indicator: projected})
        return self

    def build(self) -> pd.DataFrame:
        """Return the constructed policy scenario frame."""

        return self._scenario.sort_values(["country", "year"]).reset_index(drop=True)
