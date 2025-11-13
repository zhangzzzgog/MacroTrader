"""Forecasting with external large-language-model APIs."""
from __future__ import annotations

import importlib
import inspect
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests


_RESPONSE_VALUE_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


_DEFAULT_MODELS = {
    "openai": "gpt-5-preview",
    "anthropic": "claude-3-opus-20240229",
    "google": "gemini-1.5-pro-latest",
}


_PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


class _OpenAIAgentAdapter:
    """Thin wrapper that bridges to the openai-agent-python SDK."""

    def __init__(
        self,
        *,
        api_key: Optional[str],
        model: Optional[str],
        system_prompt: str,
        base_url: Optional[str],
        agent_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        self._runner = self._build_runner(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            base_url=base_url,
            agent_kwargs=agent_kwargs or {},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _load_sdk() -> object:
        module_candidates = [
            "openai_agent",
            "openai_agent_python",
            "openai_agent_sdk",
        ]
        last_error: Optional[ImportError] = None
        for name in module_candidates:
            try:
                return importlib.import_module(name)
            except ImportError as exc:
                last_error = exc
        raise ImportError(
            "openai-agent-python is required but not installed. Install it with "
            "`pip install openai-agent-python` and ensure it is available in the "
            "environment."
        ) from last_error

    # ------------------------------------------------------------------
    def _build_runner(
        self,
        *,
        api_key: Optional[str],
        model: Optional[str],
        system_prompt: str,
        base_url: Optional[str],
        agent_kwargs: Dict[str, object],
    ) -> Callable[[str], str]:
        sdk = self._load_sdk()

        agent_cls = None
        for attr in ("OpenAIAgent", "Agent", "LLMAgent"):
            agent_cls = getattr(sdk, attr, None)
            if inspect.isclass(agent_cls):
                break
        if agent_cls is None:
            raise ImportError(
                "openai-agent-python does not expose an Agent-compatible class."
            )

        init_params = inspect.signature(agent_cls).parameters
        init_payload = dict(agent_kwargs)
        if api_key is not None:
            for key in ("api_key", "openai_api_key", "token"):
                if key in init_params and key not in init_payload:
                    init_payload[key] = api_key
                    break
        if model is not None:
            for key in ("model", "default_model"):
                if key in init_params and key not in init_payload:
                    init_payload[key] = model
                    break
        if base_url is not None:
            for key in ("base_url", "api_base"):
                if key in init_params and key not in init_payload:
                    init_payload[key] = base_url
                    break
        for key in ("instructions", "default_instructions", "system_prompt"):
            if key in init_params and key not in init_payload:
                init_payload[key] = system_prompt
                break

        agent = agent_cls(**init_payload)
        runner: Optional[Callable[[str], object]] = None
        for attr in ("run", "invoke", "complete", "__call__"):
            candidate = getattr(agent, attr, None)
            if callable(candidate):
                runner = candidate
                break
        if runner is None:
            raise AttributeError(
                "openai-agent-python agent instance has no callable entrypoint."
            )

        def _runner(prompt: str) -> str:
            result = runner(prompt)
            if isinstance(result, str):
                return result
            if isinstance(result, dict):
                for key in ("text", "output", "message", "response"):
                    value = result.get(key)
                    if isinstance(value, str):
                        return value
                    if isinstance(value, dict) and "text" in value:
                        maybe_text = value.get("text")
                        if isinstance(maybe_text, str):
                            return maybe_text
            text_attributes = [
                "output_text",
                "text",
                "response",
                "content",
            ]
            for attr in text_attributes:
                value = getattr(result, attr, None)
                if isinstance(value, str):
                    return value
                if isinstance(value, list):
                    fragments = []
                    for item in value:
                        if isinstance(item, str):
                            fragments.append(item)
                        elif isinstance(item, dict):
                            fragment = item.get("text") or item.get("content")
                            if fragment:
                                fragments.append(str(fragment))
                        elif hasattr(item, "text"):
                            fragments.append(str(getattr(item, "text")))
                    if fragments:
                        return "\n".join(fragments)
            raise ValueError(
                "Unable to interpret response from openai-agent-python. Received: "
                f"{result!r}"
            )

        return _runner

    # ------------------------------------------------------------------
    def request(self, prompt: str) -> str:
        return self._runner(prompt)


@dataclass
class LLMForecastClient:
    """HTTP client that supports multiple LLM providers for forecasting."""

    provider: Optional[str] = None
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.2
    timeout: float = 30.0
    max_output_tokens: int = 512
    anthropic_version: str = "2023-06-01"
    system_prompt: str = (
        "You are an investment forecasting analyst. Read the given historical "
        "company investment data and estimate the future investment amount. "
        "Respond with a single number representing millions of USD."
    )
    extra_headers: Dict[str, str] = field(default_factory=dict)
    use_openai_agent: Optional[bool] = None
    openai_agent_kwargs: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = (self.provider or os.getenv("LLM_FORECAST_PROVIDER") or "openai").lower()
        self.provider = provider

        if self.use_openai_agent is None:
            env_value = os.getenv("LLM_FORECAST_USE_OPENAI_AGENT")
            self.use_openai_agent = env_value not in {None, "0", "false", "False"}

        if self.model is None:
            self.model = _DEFAULT_MODELS.get(provider, "gpt-5-preview")

        if self.api_key is None:
            # Provider-specific env var overrides the generic fallback.
            provider_key = _PROVIDER_API_KEY_ENV.get(provider)
            candidate_keys: List[str] = []
            if provider_key:
                provider_value = os.getenv(provider_key)
                if provider_value:
                    candidate_keys.append(provider_value)
            generic_value = os.getenv("LLM_FORECAST_API_KEY")
            if generic_value:
                candidate_keys.append(generic_value)
            if candidate_keys:
                self.api_key = candidate_keys[0]

        self._openai_agent: Optional[_OpenAIAgentAdapter] = None
        if provider == "openai" and self.use_openai_agent:
            try:
                self._openai_agent = _OpenAIAgentAdapter(
                    api_key=self.api_key,
                    model=self.model,
                    system_prompt=self.system_prompt,
                    base_url=self.api_url,
                    agent_kwargs=self.openai_agent_kwargs,
                )
            except ImportError as exc:
                raise ValueError(
                    "openai-agent-python must be installed to enable agent mode. "
                    "Install it with `pip install openai-agent-python` or disable "
                    "agent mode by clearing LLM_FORECAST_USE_OPENAI_AGENT."
                ) from exc

        if self.api_url is None:
            self.api_url = os.getenv("LLM_FORECAST_API_URL")

        if self.api_url is None and self._openai_agent is None:
            if provider == "openai":
                self.api_url = "https://api.openai.com/v1/chat/completions"
            elif provider == "anthropic":
                self.api_url = "https://api.anthropic.com/v1/messages"
            elif provider == "google":
                self.api_url = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.model}:generateContent"
                )

        if provider in {"openai", "anthropic", "google"} and not self.api_key:
            raise ValueError(
                "LLMForecastClient requires an API key for provider "
                f"'{provider}'. Set the appropriate environment variable (e.g. "
                f"{_PROVIDER_API_KEY_ENV.get(provider, 'LLM_FORECAST_API_KEY')}) or "
                "pass api_key explicitly."
            )

        if self._openai_agent is None and not self.api_url:
            raise ValueError(
                "LLMForecastClient requires an API URL. Provide api_url explicitly or "
                "set LLM_FORECAST_API_URL."
            )

    # ------------------------------------------------------------------
    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.provider == "anthropic":
            headers["x-api-key"] = self.api_key or ""
            headers["anthropic-version"] = self.anthropic_version
        elif self.provider == "google":
            # Google Gemini uses API key as query parameter; do not attach bearer header.
            pass
        else:
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

        headers.update(self.extra_headers)
        return headers

    # ------------------------------------------------------------------
    def _build_payload(self, prompt: str) -> Dict[str, object]:
        if self.provider == "anthropic":
            return {
                "model": self.model,
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "system": self.system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    }
                ],
            }
        if self.provider == "google":
            return {
                "systemInstruction": {
                    "role": "system",
                    "parts": [{"text": self.system_prompt}],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt,
                            }
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_output_tokens,
                },
            }

        # Default/OpenAI-compatible payload structure
        return {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

    # ------------------------------------------------------------------
    def _extract_text(self, data: Dict[str, object]) -> str:
        provider = self.provider
        if provider == "anthropic":
            content = data.get("content") if isinstance(data, dict) else None
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    text = first.get("text")
                    if text:
                        return str(text)
        elif provider == "google":
            candidates = data.get("candidates") if isinstance(data, dict) else None
            if isinstance(candidates, list) and candidates:
                candidate = candidates[0]
                content = candidate.get("content") if isinstance(candidate, dict) else None
                parts: List[Dict[str, object]] = []
                if isinstance(content, dict):
                    maybe_parts = content.get("parts")
                    if isinstance(maybe_parts, list):
                        parts = maybe_parts
                elif isinstance(content, list):
                    parts = [item for item in content if isinstance(item, dict)]
                if parts:
                    text = parts[0].get("text")
                    if text:
                        return str(text)
        else:
            choices = data.get("choices") if isinstance(data, dict) else None
            if isinstance(choices, list) and choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            text = block.get("text") or block.get("content")
                            if text:
                                return str(text)
            # Fallback for generic JSON responses
            result = data.get("result") if isinstance(data, dict) else None
            if result:
                return str(result)

        raise ValueError(
            "LLM forecast response missing content. Received: " + json.dumps(data)
        )

    # ------------------------------------------------------------------
    def request_forecast(self, prompt: str) -> float:
        if self._openai_agent is not None:
            content = self._openai_agent.request(prompt)
        else:
            payload = self._build_payload(prompt)

            request_kwargs = {
                "json": payload,
                "timeout": self.timeout,
            }

            headers = self._build_headers()
            if headers:
                request_kwargs["headers"] = headers

            if self.provider == "google" and self.api_key:
                request_kwargs["params"] = {"key": self.api_key}

            response = requests.post(self.api_url, **request_kwargs)
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict):
                raise ValueError("Unexpected response format from LLM forecast API.")

            content = self._extract_text(data)

        match = _RESPONSE_VALUE_RE.search(content)
        if not match:
            raise ValueError(
                "Could not parse numerical prediction from LLM response: " + content
            )
        return float(match.group(1))


def _format_feature_row(features: Dict[str, object]) -> str:
    parts: List[str] = []
    for key, value in features.items():
        if pd.isna(value):
            continue
        if isinstance(value, (float, np.floating)):
            parts.append(f"{key}={float(value):.3f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


@dataclass
class InvestmentAmountForecaster:
    """LLM-backed forecaster that leverages the LLMForecastClient."""

    client: Optional[LLMForecastClient] = None
    max_history_rows: int = 40
    decimal_places: Optional[int] = None

    def __post_init__(self) -> None:
        self.client = self.client or LLMForecastClient()
        self._history: List[Dict[str, object]] = []

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Sequence[float]) -> "InvestmentAmountForecaster":
        history = X.copy()
        history = history.assign(investment_amount=np.asarray(list(y), dtype=float))
        history = history.dropna(subset=["investment_amount"])
        self._history = history.tail(self.max_history_rows).to_dict(orient="records")
        return self

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._history:
            raise RuntimeError("Forecaster must be fitted with historical data before use.")

        predictions: List[float] = []
        history_block = "\n".join(
            f"- { _format_feature_row(row) }" for row in self._history
        )

        for row in X.to_dict(orient="records"):
            prompt = (
                "Historical investment observations (most recent last):\n"
                f"{history_block}\n\n"
                "Predict the future investment amount given the following context:\n"
                f"{_format_feature_row(row)}\n\n"
                "Return only the numeric prediction."
            )
            value = self.client.request_forecast(prompt)
            if self.decimal_places is not None:
                value = round(value, self.decimal_places)
            predictions.append(value)
        return np.asarray(predictions, dtype=float)

    # ------------------------------------------------------------------
    def get_feature_names(self) -> Iterable[str]:
        return list(self._history[0].keys()) if self._history else []

    # ------------------------------------------------------------------
    def get_feature_importances(self) -> Optional[pd.Series]:
        """LLM forecasts do not expose feature importances."""

        return None
