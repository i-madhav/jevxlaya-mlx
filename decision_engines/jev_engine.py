"""Adapter for TypeSafe's Jev System One model.

API key: https://console.typesafe.ai/keys -> export as TYPESAFE_API_KEY.
Docs: https://docs.typesafe.ai/api
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import requests

from .base import DecisionEngine, DecisionResult

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

# TypeSafe's published headline pricing for Jev: input tokens are billed,
# output tokens are free (it returns a typed answer, not generated prose).
PRICE_PER_MILLION_INPUT_TOKENS_USD = 0.042


class JevEngine(DecisionEngine):
    name = "jev"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_s: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Missing TYPESAFE_API_KEY. Get a key at https://console.typesafe.ai/keys "
                "and export it as TYPESAFE_API_KEY."
            )
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def decide(self, state: str, questions: Dict[str, Any]) -> DecisionResult:
        payload = {"state": state, "model": self.model, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        t0 = time.perf_counter()
        try:
            resp = self._session.post(
                self.endpoint, json=payload, headers=headers, timeout=self.timeout_s
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return DecisionResult(
                engine=self.name, answers={}, latency_ms=latency_ms, error=str(exc)
            )

        usage = body.get("usage", {})
        input_tokens = usage.get("input_tokens", 0) or 0
        usage["cost_usd"] = input_tokens * PRICE_PER_MILLION_INPUT_TOKENS_USD / 1_000_000

        return DecisionResult(
            engine=self.name,
            answers=body.get("answers", {}),
            latency_ms=latency_ms,
            usage=usage,
        )
