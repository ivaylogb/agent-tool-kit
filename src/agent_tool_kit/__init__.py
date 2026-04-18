"""agent-tool-kit: production patterns for LLM tool design.

Most agent failures are tool failures, not prompt failures. Bad schemas cause
hallucinated parameters; thin wrappers create multi-step fragility; verbose
outputs blow the context window; missing scope restrictions cause misrouting;
unstructured errors leave the model unable to recover.

This package provides the building blocks to design tools that absorb the
variance of a stochastic caller and return structured feedback that teaches
the model how to recover.

Public surface:
    - ``tool`` decorator + ``Tool`` class: schema-validated, observable,
      idempotency-aware tool wrappers.
    - ``ToolError`` / ``ErrorCategory``: structured errors the LLM can reason about.
    - ``CapabilityRegistry``: progressive disclosure for tool catalogues at scale.
    - ``ToolClassifier``: Haiku-backed router that selects relevant tools per task.
    - ``AuditLog`` / ``ToolCallRecord``: append-only audit trail with replay.
    - ``IdempotencyCache``: pluggable backend for safely re-callable actions.
    - ``RetryPolicy`` / ``CircuitBreaker`` / ``defensive_call``: defensive wrappers
      for unreliable upstream services.
    - ``project`` / ``highlight_fragments``: context-compression helpers.
"""

from agent_tool_kit.base import Tool, refuse, tool
from agent_tool_kit.classifier import ToolClassifier
from agent_tool_kit.compression import (
    highlight_fragments,
    project,
    project_many,
    truncate_text,
)
from agent_tool_kit.defensive import (
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    defensive_call,
    with_retries,
    with_timeout,
)
from agent_tool_kit.errors import ErrorCategory, ToolError, ToolException
from agent_tool_kit.idempotency import IdempotencyCache
from agent_tool_kit.observability import AuditLog, ToolCallRecord
from agent_tool_kit.registry import CapabilityRegistry

__all__ = [
    "AuditLog",
    "CapabilityRegistry",
    "CircuitBreaker",
    "CircuitState",
    "ErrorCategory",
    "IdempotencyCache",
    "RetryPolicy",
    "Tool",
    "ToolCallRecord",
    "ToolClassifier",
    "ToolError",
    "ToolException",
    "defensive_call",
    "highlight_fragments",
    "project",
    "project_many",
    "refuse",
    "tool",
    "truncate_text",
    "with_retries",
    "with_timeout",
]

__version__ = "0.1.0"
