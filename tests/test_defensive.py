"""Tests for retries, timeouts, circuit breaker, and the composed defensive_call."""

from __future__ import annotations

import time

import pytest

from agent_tool_kit import (
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    ToolException,
    defensive_call,
    with_retries,
    with_timeout,
)
from agent_tool_kit.errors import ErrorCategory

# ----- RetryPolicy ---------------------------------------------------------


def test_with_retries_returns_after_eventual_success():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    result = with_retries(fn, RetryPolicy(max_attempts=5, base_delay=0, jitter=0))
    assert result == "ok"
    assert calls["n"] == 3


def test_with_retries_reraises_after_exhaustion():
    def fn():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        with_retries(fn, RetryPolicy(max_attempts=2, base_delay=0, jitter=0))


def test_with_retries_respects_retry_on_filter():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise TypeError("non-retryable")

    with pytest.raises(TypeError):
        with_retries(
            fn,
            RetryPolicy(max_attempts=3, base_delay=0, jitter=0, retry_on=(ValueError,)),
        )
    assert calls["n"] == 1  # didn't retry


def test_retry_policy_delay_grows_exponentially():
    p = RetryPolicy(base_delay=1.0, multiplier=2.0, jitter=0.0, max_delay=10.0)
    assert p.delay_for(1) == 1.0
    assert p.delay_for(2) == 2.0
    assert p.delay_for(3) == 4.0
    assert p.delay_for(10) == 10.0  # capped


# ----- with_timeout --------------------------------------------------------


def test_with_timeout_returns_fast_result():
    assert with_timeout(lambda: "ok", 1.0) == "ok"


def test_with_timeout_raises_on_slow_call():
    def slow():
        time.sleep(0.5)
        return "too late"

    with pytest.raises(TimeoutError):
        with_timeout(slow, 0.1)


def test_with_timeout_propagates_exceptions():
    def fn():
        raise ValueError("inner")

    with pytest.raises(ValueError):
        with_timeout(fn, 1.0)


# ----- CircuitBreaker ------------------------------------------------------


def test_breaker_opens_after_threshold():
    b = CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0)
    assert b.state is CircuitState.CLOSED
    b.record_failure()
    assert b.state is CircuitState.CLOSED
    b.record_failure()
    assert b.state is CircuitState.OPEN
    assert b.allow() is False


def test_breaker_half_opens_after_cooldown():
    fake_time = {"now": 0.0}
    b = CircuitBreaker(
        failure_threshold=1,
        cooldown_seconds=5.0,
        clock=lambda: fake_time["now"],
    )
    b.record_failure()
    assert b.state is CircuitState.OPEN
    fake_time["now"] = 6.0
    assert b.state is CircuitState.HALF_OPEN
    # First trial allowed, second blocked.
    assert b.allow() is True
    assert b.allow() is False
    # Success closes.
    b.record_success()
    assert b.state is CircuitState.CLOSED


def test_breaker_failure_in_half_open_reopens():
    fake_time = {"now": 0.0}
    b = CircuitBreaker(
        failure_threshold=1,
        cooldown_seconds=5.0,
        clock=lambda: fake_time["now"],
    )
    b.record_failure()
    fake_time["now"] = 6.0
    assert b.allow() is True  # half-open trial reserved
    b.record_failure()
    assert b.state is CircuitState.OPEN


# ----- defensive_call ------------------------------------------------------


def test_defensive_call_returns_value_on_success():
    assert defensive_call(lambda: "ok") == "ok"


def test_defensive_call_converts_exception_to_tool_exception():
    def fn():
        raise ValueError("upstream")

    with pytest.raises(ToolException) as exc:
        defensive_call(fn)
    assert exc.value.error.category is ErrorCategory.UPSTREAM_FAILURE


def test_defensive_call_converts_timeout():
    def fn():
        time.sleep(0.5)

    with pytest.raises(ToolException) as exc:
        defensive_call(fn, timeout_seconds=0.1)
    assert exc.value.error.category is ErrorCategory.TIMEOUT
    assert exc.value.error.retryable is True


def test_defensive_call_blocks_when_breaker_open():
    b = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
    b.record_failure()  # opens
    with pytest.raises(ToolException) as exc:
        defensive_call(lambda: "ok", circuit_breaker=b)
    assert exc.value.error.category is ErrorCategory.CIRCUIT_OPEN


def test_defensive_call_records_success_to_breaker():
    b = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)
    b.record_failure()
    defensive_call(lambda: "ok", circuit_breaker=b)
    assert b.state is CircuitState.CLOSED


def test_defensive_call_with_retries_succeeds_eventually():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return "ok"

    result = defensive_call(
        fn,
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0, jitter=0),
    )
    assert result == "ok"
    assert calls["n"] == 2
