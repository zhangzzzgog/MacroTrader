# MacroTrader

MacroTrader provides tooling to build company-level intelligent agents that use
historical investment behaviour together with macro-policy indicators to
forecast future outbound investment flows.

## Features

- Data validation helpers for investment transactions and policy indicator
  tables.
- Feature engineering that captures temporal investment patterns, lagged
  allocations, and policy context from both the home and target countries.
- A scikit-learn based forecasting pipeline with automated preprocessing for
  numerical and categorical variables.
- An agent abstraction that trains a dedicated model for each company and can
  predict investment amounts for specific target countries over 3–5 year
  horizons (or other custom horizons).
- Scenario building utilities to craft custom policy trajectories and stress
  tests.

## Installation

```bash
pip install -r requirements.txt
```

## Data requirements

Investment data should contain at least the following columns:

- `company_name`
- `company_country`
- `target_country`
- `investment_date`
- `investment_amount`

Optional columns such as `jobs_created` are automatically incorporated when
present. Policy datasets require a `country` column, a `year` column, and any
number of numerical policy indicators.

## Usage example

```python
import pandas as pd

from macro_trader import (
    CompanyInvestmentAgent,
    FeatureBuilder,
    InvestmentAmountForecaster,
    PolicyScenarioBuilder,
)

# Load your historical data
investments = pd.read_csv("investments.csv")
policy = pd.read_csv("policy_indicators.csv")

# Train an agent for a single company
agent = CompanyInvestmentAgent(
    company_name="Global Manufacturing Ltd",
    feature_builder=FeatureBuilder(),
    forecaster=InvestmentAmountForecaster(),
)
agent.fit(investments, policy)

# Build a forward-looking policy scenario
scenario_builder = PolicyScenarioBuilder(policy)
scenario = (
    scenario_builder
    .apply_growth_projection(
        country="Targetland",
        indicator="tax_incentive_index",
        annual_growth=0.02,
        years=[2024, 2025, 2026, 2027, 2028],
    )
    .build()
)

# Predict investment volumes for the next 3–5 years
predictions = agent.predict(
    target_country="Targetland",
    policy_scenario=scenario,
    years_ahead=(3, 4, 5),
)
print(predictions)
```

The resulting dataframe contains the predicted investment amounts per forecast
year. Supply different policy scenarios to evaluate how macro changes influence
corporate investment trajectories.
