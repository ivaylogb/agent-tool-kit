"""Defensive wrappers for tools that call unreliable upstream services.

Three composable primitives:

- ``RetryPolicy`` — exponential backoff with jitter for transient failures.
- ``CircuitBreaker`` — fails fast after a threshold of consecutive errors.
- ``with_timeout`` — wall-clock cap on a synchronous call.

``defensive_call`` composes all three and converts the failure modes the agent
should reason about (timeout, circuit open, exhausted retries) into
``ToolException`` carrying a structured ``ToolError``.

Design note: timeouts use a worker thread because the toolkit targets
synchronous handlers. The thread continues running on timeout — callers
treat upstream calls as best-effort and accept that one orphan thread per
timeout is the cost of not requiring async everywhere.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from agent_tool_kit.errors import ErrorCategory, ToolError, ToolException


@dataclass
class RetryPolicy:
    """Exponential backoff with optional jitter.

    ``retry_on`` is a tuple of exception types that count as transient. By
    default everything counts; narrow it (e.g., to ``(ConnectionError,
    TimeoutError)``) to avoid retrying programmer errors.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    multiplier: float = 2.0
    jitter: float = 0.25
    retry_on: tuple[type[BaseException], ...] = (Exception,)

    def delay_for(self, attempt: int) -> float:
        """Compute the delay before the next attempt (1-indexed)."""
        if attempt < 1:
            return 0.0
        d = min(self.max_delay, self.base_delay * (self.multiplier ** (attempt - 1)))
        if self.jitter:
            d = d * (1.0 + random.uniform(-self.jitter, self.jitter))
        return max(0.0, d)


class CircuitState(str, Enum):
    CLOSED = "closed"        # All calls flow through.
    OPEN = "open"            # Calls fail fast until cooldown elapses.
    HALF_OPEN = "half_open"  # One trial call allowed; success closes, failure re-opens.


class CircuitBreaker:
    """Thread-safe circuit breaker.

    Default policy: 5 consecutive **outcomes recorded as failures** open the
    breaker for 30 seconds. During the open window all calls fail fast. After
    the cooldown, one trial call is allowed; if it succeeds the breaker
    closes, if it fails the cooldown restarts.

    Granularity note: when paired with ``defensive_call``, the unit of
    failure is one *invocation* of ``defensive_call``, not one upstream
    attempt. A defensive call configured with ``RetryPolicy(max_attempts=3)``
    that exhausts all 3 retries records exactly **one** failure on the
    breaker — the breaker sees user-visible outcomes, not internal retries.
    This matches the behavior of production breakers (Hystrix-style command
    semantics) and prevents retried-then-recovered calls from inappropriately
    moving the breaker toward open. If you want each upstream attempt to
    count, omit the retry policy and let the caller drive retries.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 1,
        clock: Callable[[], float] = time.time,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_to_half_open_locked()
            return self._state

    def _maybe_transition_to_half_open_locked(self) -> None:
        if (
            self._state is CircuitState.OPEN
            and self._opened_at is not None
            and self._clock() - self._opened_at >= self.cooldown_seconds
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0

    def allow(self) -> bool:
        """Return True if a call is permitted; reserves a half-open slot if applicable."""
        with self._lock:
            self._maybe_transition_to_half_open_locked()
            if self._state is CircuitState.OPEN:
                return False
            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    return False
                self._half_open_calls += 1
            return True

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._opened_at = None
            self._half_open_calls = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state is CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._half_open_calls = 0
            elif self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()


def with_retries(
    fn: Callable[[], Any],
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Run ``fn`` up to ``policy.max_attempts`` times. Returns the result, or
    re-raises the last exception if all attempts fail.
    """
    if policy.max_attempts < 1:
        raise ValueError("RetryPolicy.max_attempts must be >= 1")
    last_exc: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except policy.retry_on as e:
            last_exc = e
            if attempt >= policy.max_attempts:
                break
            sleep(policy.delay_for(attempt))
    assert last_exc is not None
    raise last_exc


def with_timeout(fn: Callable[[], Any], timeout_seconds: float) -> Any:
    """Run ``fn`` in a worker thread; raise TimeoutError if it doesn't finish in time.

    The worker continues running after a timeout — there's no safe way to
    interrupt arbitrary synchronous code in Python. Callers should treat
    the upstream operation as having an unknown final state.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    result: list[Any] = [None]
    error: list[BaseException | None] = [None]
    done = threading.Event()

    def runner() -> None:
        try:
            result[0] = fn()
        except BaseException as e:  # noqa: BLE001 — propagate everything to caller
            error[0] = e
        finally:
            done.set()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    if not done.wait(timeout_seconds):
        raise TimeoutError(f"Call exceeded {timeout_seconds}s")
    if error[0] is not None:
        raise error[0]
    return result[0]


def defensive_call(
    fn: Callable[[], Any],
    *,
    retry_policy: RetryPolicy | None = None,
    timeout_seconds: float | None = None,
    circuit_breaker: CircuitBreaker | None = None,
) -> Any:
    """Run ``fn`` with optional retry, timeout, and circuit-breaker protection.

    Failure modes are converted to ``ToolException`` carrying a typed
    ``ToolError`` — the agent reasons about the category to choose its
    next move. Successful results are passed through unchanged.

    Breaker accounting: the breaker (if provided) sees **one outcome per
    defensive_call invocation**, not one per retry attempt. If
    ``retry_policy`` succeeds on the second attempt, the breaker records a
    success. If retries are exhausted, the breaker records exactly one
    failure regardless of how many internal attempts ran. This is intentional
    — it matches Hystrix-style command semantics and means a
    ``CircuitBreaker(failure_threshold=N)`` opens after N consecutive
    *user-visible* failures, not N internal attempts. See ``CircuitBreaker``
    for the rationale.
    """
    if circuit_breaker is not None and not circuit_breaker.allow():
        raise ToolException(
            ToolError(
                category=ErrorCategory.CIRCUIT_OPEN,
                message="Upstream service is unavailable; circuit breaker is open.",
                retryable=True,
                suggested_action=(
                    "Wait for the cooldown to elapse or fall back to a different action."
                ),
            )
        )

    def attempt() -> Any:
        if timeout_seconds is not None:
            return with_timeout(fn, timeout_seconds)
        return fn()

    try:
        result = with_retries(attempt, retry_policy) if retry_policy is not None else attempt()
    except TimeoutError as e:
        if circuit_breaker is not None:
            circuit_breaker.record_failure()
        raise ToolException(
            ToolError(
                category=ErrorCategory.TIMEOUT,
                message=str(e) or "Upstream call timed out.",
                retryable=True,
                suggested_action="Retry the request; the upstream call exceeded the time budget.",
            )
        ) from e
    except ToolException:
        if circuit_breaker is not None:
            circuit_breaker.record_failure()
        raise
    except Exception as e:
        if circuit_breaker is not None:
            circuit_breaker.record_failure()
        raise ToolException(
            ToolError(
                category=ErrorCategory.UPSTREAM_FAILURE,
                message=f"Upstream call failed: {type(e).__name__}: {e}",
                retryable=False,
                suggested_action="Inspect the error and decide whether to surface it or escalate.",
                details={"exception_type": type(e).__name__},
            )
        ) from e

    if circuit_breaker is not None:
        circuit_breaker.record_success()
    return result
