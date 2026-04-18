"""A controllable fake "weather API" for the defensive-wrapping example.

The real API is whatever flaky upstream you actually have to call. This stand-in
lets the tool demonstrate retries, timeouts, and the circuit breaker without
network calls — and lets the unit tests deterministically reproduce the failure
modes.

``FakeWeatherAPI`` is configurable per-test:
    api = FakeWeatherAPI(failure_pattern=["error", "timeout", "ok"])
    api.fetch("Berlin")  # raises
    api.fetch("Berlin")  # times out
    api.fetch("Berlin")  # returns the payload
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any


class UpstreamError(RuntimeError):
    """Stand-in for whatever exception the real upstream client raises."""


class FakeWeatherAPI:
    """Configurable fake. Failure pattern is consumed in order, then loops."""

    def __init__(
        self,
        failure_pattern: Sequence[str] = ("ok",),
        slow_seconds: float = 2.0,
    ):
        self._pattern = list(failure_pattern)
        self._index = 0
        self.slow_seconds = slow_seconds
        self.call_count = 0

    def fetch(self, city: str) -> dict[str, Any]:
        self.call_count += 1
        if not self._pattern:
            outcome = "ok"
        else:
            outcome = self._pattern[self._index % len(self._pattern)]
            self._index += 1
        if outcome == "error":
            raise UpstreamError(f"upstream failed for {city!r}")
        if outcome == "timeout":
            time.sleep(self.slow_seconds)
            return _payload(city)
        if outcome == "rate_limit":
            raise UpstreamError(f"upstream returned 429 for {city!r}")
        return _payload(city)


def _payload(city: str) -> dict[str, Any]:
    return {
        "city": city,
        "temperature_c": 21,
        "conditions": "partly cloudy",
        "humidity_pct": 58,
        "wind_kmh": 14,
        "observed_at": "2026-04-17T12:00:00Z",
    }
