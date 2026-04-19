"""External-API wrapper tool — defensive composition pattern.

A typical anti-pattern: a tool calls the upstream API directly, lets exceptions
escape into the agent loop, and provides no retry or circuit protection. The
LLM either crashes the conversation or invents a recovery story.

This module shows the production composition:

- ``RetryPolicy`` for transient failures (jittered exponential backoff)
- ``with_timeout`` to bound wall-clock cost of slow upstreams
- ``CircuitBreaker`` to fail fast once an upstream is clearly down
- ``defensive_call`` to compose the above and convert failures to typed errors
- ``IdempotencyCache`` so a duplicate call within a session returns the cached
  reading instead of re-paying for the upstream round-trip

The tool factory accepts an injected ``api`` so tests can plug in
``FakeWeatherAPI`` with a deterministic failure pattern.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field

from agent_tool_kit import (
    AuditLog,
    CapabilityRegistry,
    CircuitBreaker,
    IdempotencyCache,
    RetryPolicy,
    Tool,
    defensive_call,
    project,
    tool,
)

from .fake_api import FakeWeatherAPI


class WeatherInput(BaseModel):
    city: str = Field(
        description="City name as a free-form string. Example: 'Berlin'.",
        min_length=1,
        max_length=80,
    )


def build_weather_tool(
    *,
    api: FakeWeatherAPI | None = None,
    audit_log: AuditLog | None = None,
    cache: IdempotencyCache | None = None,
    retry_policy: RetryPolicy | None = None,
    breaker: CircuitBreaker | None = None,
    timeout_seconds: float = 1.0,
) -> tuple[Tool, dict[str, Any]]:
    """Build the weather tool and return it alongside its wired dependencies.

    Returning the deps lets a caller inspect/reset the breaker, dump audit
    metrics, or share the cache across multiple tools without reaching into
    the closure.
    """
    # Use ``is not None`` rather than ``or`` because IdempotencyCache and AuditLog
    # define ``__len__``, so an empty instance is falsy under truthy-coercion and
    # would be silently replaced with a fresh one — discarding the caller's choice.
    api = api if api is not None else FakeWeatherAPI()
    audit_log = audit_log if audit_log is not None else AuditLog()
    cache = cache if cache is not None else IdempotencyCache()
    retry_policy = retry_policy if retry_policy is not None else RetryPolicy(
        max_attempts=3,
        base_delay=0.05,
        max_delay=0.2,
        multiplier=2.0,
        jitter=0.0,
    )
    breaker = breaker if breaker is not None else CircuitBreaker(
        failure_threshold=3, cooldown_seconds=2.0,
    )

    @tool(
        name="get_current_weather",
        description=(
            "Fetch current weather for a city via the external weather API. "
            "Wrapped with retries (3 attempts, exponential backoff), a 1s "
            "timeout, and a circuit breaker that opens after 3 consecutive "
            "failures and cools down for 2s. Idempotent within a session: "
            "duplicate calls for the same city return the cached reading."
        ),
        when_not_to_use=(
            "Do not call for forecasts beyond 'right now' — this returns the "
            "current observation only. Do not use to look up historical "
            "weather. Do not retry manually if you receive a circuit_open "
            "error; the breaker is already enforcing the cooldown."
        ),
        input_model=WeatherInput,
        tags=["weather", "external"],
        audit_log=audit_log,
        idempotent=True,
        idempotency_cache=cache,
    )
    def get_current_weather(city: str) -> dict[str, Any]:
        raw = defensive_call(
            lambda: api.fetch(city),
            retry_policy=retry_policy,
            timeout_seconds=timeout_seconds,
            circuit_breaker=breaker,
        )
        # Project away any internal upstream fields the LLM doesn't need.
        return project(
            raw,
            ["city", "temperature_c", "conditions", "humidity_pct", "wind_kmh", "observed_at"],
        )

    deps = {
        "api": api,
        "audit_log": audit_log,
        "cache": cache,
        "retry_policy": retry_policy,
        "breaker": breaker,
    }
    return get_current_weather, deps


def build_registry(
    api_factory: Callable[[], FakeWeatherAPI] | None = None,
) -> CapabilityRegistry:
    """Construct a registry containing the weather tool with fresh dependencies.

    Symmetric with the other examples' ``build_registry`` — returns just the
    registry. If you need to introspect the breaker state or audit log,
    construct the tool directly via ``build_weather_tool`` instead.
    """
    api = (api_factory or FakeWeatherAPI)()
    weather, _deps = build_weather_tool(api=api)
    return CapabilityRegistry([weather])
