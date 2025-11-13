"""Data model definitions for MacroTrader."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping, Optional


@dataclass(frozen=True)
class InvestmentRecord:
    """Represents an investment made by a company into a target country."""

    company_name: str
    company_country: str
    target_country: str
    investment_date: date
    investment_amount: float
    jobs_created: Optional[int] = None
    metadata: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class PolicyRecord:
    """Represents policy information for a country in a specific year."""

    country: str
    year: int
    indicators: Mapping[str, float]
