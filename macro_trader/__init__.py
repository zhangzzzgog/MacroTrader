"""MacroTrader package.

Provides tooling to build company-level investment prediction agents that
combine corporate investment histories with macro policy information.
"""

from .agent import CompanyInvestmentAgent
from .feature_engineering import FeatureBuilder
from .forecasting import InvestmentAmountForecaster, LLMForecastClient
from .scenarios import PolicyScenarioBuilder

__all__ = [
    "CompanyInvestmentAgent",
    "FeatureBuilder",
    "InvestmentAmountForecaster",
    "LLMForecastClient",
    "PolicyScenarioBuilder",
]
