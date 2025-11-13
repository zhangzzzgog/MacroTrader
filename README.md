# MacroTrader

MacroTrader provides tooling to build company-level intelligent agents that use
historical investment behaviour together with macro-policy indicators to
forecast future outbound investment flows.

## Features

- Data validation helpers for investment transactions and policy indicator
  tables.
- Feature engineering that captures temporal investment patterns, lagged
  allocations, and policy context from both the home and target countries.
- An LLM-powered forecasting client that calls an external API to infer future
  investment amounts from engineered features.
- An agent abstraction that trains a dedicated model for each company and can
  predict investment amounts for specific target countries over 3–5 year
  horizons (or other custom horizons).
- Scenario building utilities to craft custom policy trajectories and stress
  tests.

## Installation

```bash
pip install -r requirements.txt
```

### Configure your LLM endpoint

`InvestmentAmountForecaster` can call OpenAI GPT-5, Anthropic Claude 3, or
Google Gemini models out of the box. Choose a provider via the
`LLM_FORECAST_PROVIDER` environment variable (defaults to `openai`) and supply
the matching API key:

| Provider  | `LLM_FORECAST_PROVIDER` | Default model                | Required key             |
|-----------|-------------------------|------------------------------|--------------------------|
| OpenAI    | `openai`                | `gpt-5-preview`              | `OPENAI_API_KEY`         |
| Anthropic | `anthropic`             | `claude-3-opus-20240229`     | `ANTHROPIC_API_KEY`      |
| Google    | `google`                | `gemini-1.5-pro-latest`      | `GOOGLE_API_KEY`         |

Set the provider-specific key in your environment before running forecasts. The
client automatically selects the correct REST endpoint for the chosen provider.

If you prefer to use OpenAI's Agent SDK (`openai-agent-python`) instead of the
raw REST endpoint, install the package and enable agent mode:

```bash
pip install openai-agent-python
export OPENAI_API_KEY="sk-..."
export LLM_FORECAST_USE_OPENAI_AGENT=1
```

Optional keyword arguments for the agent constructor can be provided via the
`openai_agent_kwargs` argument when instantiating `LLMForecastClient`.

To override any defaults (for example, to target Azure OpenAI or a private
gateway), set `LLM_FORECAST_API_URL`, `LLM_FORECAST_API_KEY`, or pass custom
values directly when instantiating `LLMForecastClient`.

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

# Sample experiment

An end-to-end toy example is provided in `examples/sample_company_experiment.py`.
Run it to generate a synthetic history for a fictional manufacturer, train an
agent, and produce 3–5 year forecasts for "Targetland":

```bash
python examples/sample_company_experiment.py
```

> **Note:** The script calls the configured LLM endpoint for every prediction.
> Set `LLM_FORECAST_PROVIDER` (if you want a provider other than OpenAI) and the
> matching API key before running it—for example `OPENAI_API_KEY`,
> `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`.

The script prints a forecast table for the configured company and policy
scenario, which you can adapt to plug in your own datasets and assumptions.

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
