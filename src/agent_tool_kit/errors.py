"""Structured tool errors.

The contract: tool handlers never let exceptions escape into the agent loop.
Failures become ``ToolError`` envelopes — typed objects the LLM can inspect
and reason about (retry on transient, ask the user for missing info, or escalate).

Why this matters: a Python traceback in the context window is worse than useless.
It wastes tokens and the model can't act on it. A structured error like
``{"category": "outside_window", "retryable": false, "suggested_action": "..."}``
gives the model exactly what it needs to choose its next step.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCategory(str, Enum):
    """Standard error categories. Domain-specific tools may extend by passing
    a ``str``-coercible value to ``ToolError`` directly via ``model_construct``,
    but the default categories cover the dispositions the LLM most often needs
    to distinguish.
    """

    INVALID_INPUT = "invalid_input"            # Schema or semantic validation failed
    NOT_FOUND = "not_found"                    # Resource doesn't exist
    UNAUTHORIZED = "unauthorized"              # Caller lacks permission
    PRECONDITION_FAILED = "precondition_failed"  # Resource exists but is in wrong state
    TIMEOUT = "timeout"                        # Upstream call exceeded the time budget
    UPSTREAM_FAILURE = "upstream_failure"      # Upstream returned a non-success
    RATE_LIMITED = "rate_limited"              # Quota exhausted; back off
    CIRCUIT_OPEN = "circuit_open"              # Breaker tripped; fail fast
    CONFLICT = "conflict"                      # State changed under us
    INTERNAL = "internal"                      # Unexpected handler bug
    REFUSAL = "refusal"                        # Tool deliberately declined


class ToolError(BaseModel):
    """Structured error envelope a tool returns instead of raising.

    The shape is deliberately small: the LLM should be able to inspect
    ``category`` and ``retryable`` and ``suggested_action`` to decide what
    to do next. ``details`` is escape hatch for diagnostics — keep it small,
    since it lands in the context window.
    """

    category: ErrorCategory
    message: str
    retryable: bool = False
    suggested_action: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    def to_response(self) -> dict[str, Any]:
        """Wrap the error in the ``{"error": {...}}`` envelope the agent sees."""
        return {"error": self.model_dump(mode="json", exclude_none=True)}


class ToolException(Exception):
    """Internal exception a handler may raise to short-circuit with a structured error.

    Caught by the ``Tool`` wrapper and converted to the ``{"error": {...}}``
    envelope. Handler authors who prefer raising over returning errors can use:

        raise ToolException(ToolError(
            category=ErrorCategory.NOT_FOUND,
            message=f"Order {order_id!r} not found.",
            retryable=False,
        ))

    Both styles produce the same wire format. Returning is more explicit;
    raising is more ergonomic for nested helpers.
    """

    def __init__(self, error: ToolError):
        super().__init__(error.message)
        self.error = error
