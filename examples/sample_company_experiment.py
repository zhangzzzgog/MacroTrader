"""Run a sample experiment for a single company's investment forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_trader import (
    CompanyInvestmentAgent,
    FeatureBuilder,
    InvestmentAmountForecaster,
    PolicyScenarioBuilder,
)


def build_sample_investment_data() -> pd.DataFrame:
    """Create a toy investment history for a fictional manufacturing firm."""

    records = []
    rng = np.random.default_rng(42)
    base_year = 2014

    # Generate annual investments towards two target countries
    for i, target in enumerate(["Targetland", "Alliedonia"], start=1):
        trend = 50 + i * 5
        for year_offset in range(0, 10):
            year = base_year + year_offset
            amount = trend * (1 + 0.07 * year_offset) + rng.normal(0, 5)
            jobs = 100 + i * 15 + rng.integers(-10, 10)
            records.append(
                {
                    "company_name": "Alpha Tech Manufacturing",
                    "company_country": "Homeland",
                    "target_country": target,
                    "investment_date": pd.Timestamp(year, 12, 31),
                    "investment_amount": round(max(amount, 5.0), 2),
                    "jobs_created": int(max(jobs, 0)),
                }
            )

    investment_df = pd.DataFrame(records).sort_values("investment_date").reset_index(drop=True)
    return investment_df


def build_sample_policy_data() -> pd.DataFrame:
    """Return fabricated policy indicators for home and target countries."""

    years = list(range(2014, 2024))
    policy_rows = []

    for country in ["Homeland", "Targetland", "Alliedonia"]:
        for year in years:
            base = 70 if country == "Homeland" else 50
            adjustment = 5 if country == "Targetland" else 0
            policy_rows.append(
                {
                    "country": country,
                    "year": year,
                    "tax_incentive_index": base + adjustment + 0.5 * (year - 2014),
                    "ease_of_business": 60 + adjustment + 0.3 * (year - 2014),
                }
            )

    return pd.DataFrame(policy_rows)


def run_experiment() -> pd.DataFrame:
    """Fit the agent and return future investment predictions."""

    investments = build_sample_investment_data()
    policy = build_sample_policy_data()

    try:
        forecaster = InvestmentAmountForecaster()
    except ValueError as exc:
        raise RuntimeError(
            "InvestmentAmountForecaster configuration error. Ensure the "
            "appropriate API credentials are set (e.g. OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY, or GOOGLE_API_KEY). For OpenAI agent mode, "
            "install openai-agent-python and export "
            "LLM_FORECAST_USE_OPENAI_AGENT=1. Otherwise supply a custom "
            "LLM_FORECAST_API_URL and key."
        ) from exc

    agent = CompanyInvestmentAgent(
        company_name="Alpha Tech Manufacturing",
        feature_builder=FeatureBuilder(),
        forecaster=forecaster,
    )
    agent.fit(investments, policy)

    scenario_builder = PolicyScenarioBuilder(policy)

    forward_years = [2024, 2025, 2026, 2027, 2028]
    scenario_builder.apply_growth_projection(
        country="Targetland",
        indicator="tax_incentive_index",
        annual_growth=0.03,
        years=forward_years,
    )
    scenario_builder.apply_growth_projection(
        country="Targetland",
        indicator="ease_of_business",
        annual_growth=0.015,
        years=forward_years,
    )

    # Maintain home-country indicators at the latest observed value.
    latest_home = (
        policy[policy["country"] == "Homeland"]
        .sort_values("year")
        .iloc[-1]
        .to_dict()
    )
    for year in forward_years:
        scenario_builder.set_indicator_values(
            country="Homeland",
            year=year,
            indicators={
                "tax_incentive_index": latest_home["tax_incentive_index"],
                "ease_of_business": latest_home["ease_of_business"],
            },
        )

    scenario = scenario_builder.build()

    predictions = agent.predict(
        target_country="Targetland",
        policy_scenario=scenario,
        years_ahead=(3, 4, 5),
    )
    return predictions


def main() -> None:
    predictions = run_experiment()
    print("Sample investment forecasts for Alpha Tech Manufacturing (Targetland):")
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
