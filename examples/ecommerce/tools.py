"""E-commerce tools — the fat tool pattern.

Three things to study here:

1. **Schema-as-feedback** — every input field has a ``Field(description=...)``
   the LLM reads. Bad inputs produce structured ``invalid_input`` errors that
   tell the model exactly which field to fix.

2. **Scope restrictions** — every ``@tool`` carries a ``when_not_to_use`` block.
   Without it, models call the most semantically similar tool regardless of
   correctness. With it, misrouting drops sharply.

3. **Fat tool composition** — ``process_return_request`` wraps the chain of
   lookup → eligibility check → initiate return into one call. The LLM can't
   forget a step, can't reorder them, and can't decide on its own that the
   return window doesn't apply.

The five thin tools (``lookup_order``, ``get_tracking_status``,
``initiate_return``, ``cancel_order``, ``escalate_to_human``) match the
schema agent-eval-loop's customer_support example expects, so
``get_handlers()`` is a drop-in replacement for that example's ``mocks.py``.

State scoping note: each call to ``build_registry()`` (or ``get_handlers()``)
constructs a fresh ``AuditLog`` and ``IdempotencyCache``, so two registries
never share state. Pass your own instances if you want to share across
multiple registries within a session.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_tool_kit import (
    AuditLog,
    CapabilityRegistry,
    ErrorCategory,
    IdempotencyCache,
    Tool,
    ToolError,
    ToolException,
    tool,
)

from .data import ORDERS, days_since_order, tracking_payload

RETURN_WINDOW_DAYS = 30


# -- Pydantic input models ----------------------------------------------------

class OrderIdInput(BaseModel):
    order_id: str = Field(
        description="The order number, formatted as 'ORD-XXXXX' (e.g., 'ORD-78234').",
        pattern=r"^ORD-\d+$",
    )


class ReturnInput(BaseModel):
    order_id: str = Field(
        description="The order number, formatted as 'ORD-XXXXX'.",
        pattern=r"^ORD-\d+$",
    )
    reason: str = Field(
        description=(
            "Why the customer is returning. Examples: 'wrong size', "
            "'defective', 'changed mind'."
        ),
        min_length=2,
    )


class CancelInput(BaseModel):
    order_id: str = Field(
        description="The order number, formatted as 'ORD-XXXXX'.",
        pattern=r"^ORD-\d+$",
    )
    reason: str = Field(
        description="Customer's reason for cancellation.",
        min_length=2,
    )


class EscalateInput(BaseModel):
    reason: str = Field(
        description=(
            "Why this needs human attention. Be specific — this is the human "
            "agent's first signal about the case."
        ),
        min_length=4,
    )
    priority: Literal["normal", "high", "urgent"] = Field(
        description=(
            "normal: standard queue. high: frustrated customer or repeat "
            "contact. urgent: safety concern or potential legal issue."
        ),
    )
    context_summary: str = Field(
        description=(
            "Brief summary for the human agent: what the customer needs, "
            "what you've tried, what data you've retrieved. The human should "
            "NOT have to re-ask the customer for information you already have."
        ),
        min_length=10,
    )


# -- Tool factories -----------------------------------------------------------
#
# Each builder closure-captures the audit log and idempotency cache so that
# callers of ``build_registry`` get fresh state per registry. Tests get
# isolation; long-running services share a single audit log by passing it in.


def _make_lookup_order(audit: AuditLog) -> Tool:
    @tool(
        name="lookup_order",
        description=(
            "Look up an order by order number. Returns order details including "
            "items, status, dates, and payment summary.\n\n"
            "WHEN TO USE: This should be your FIRST tool call in almost every "
            "conversation. Call it as soon as you have an order number."
        ),
        when_not_to_use=(
            "Do not call this tool to get tracking/shipping details — use "
            "get_tracking_status. Do not guess order numbers if the customer "
            "hasn't provided one — ask first."
        ),
        input_model=OrderIdInput,
        tags=["orders", "ecommerce"],
        audit_log=audit,
    )
    def lookup_order(order_id: str) -> dict[str, Any]:
        order = ORDERS.get(order_id)
        if order is None:
            raise ToolException(ToolError(
                category=ErrorCategory.NOT_FOUND,
                message=f"No order with id {order_id!r} exists.",
                retryable=False,
                suggested_action=(
                    "Ask the customer to double-check their order number "
                    "against their confirmation email."
                ),
            ))
        return dict(order)

    return lookup_order


def _make_get_tracking_status(audit: AuditLog) -> Tool:
    @tool(
        name="get_tracking_status",
        description=(
            "Get real-time tracking and shipping status for an order.\n\n"
            "WHEN TO USE: After confirming via lookup_order that the order's "
            "status is 'shipped' or 'delivered'. This is the tool for "
            "'where is my package?' questions."
        ),
        when_not_to_use=(
            "Do not call for orders with status 'pending' or 'processing' — "
            "they haven't shipped yet, the call will fail. Do not use this tool "
            "to check order details (items, price) — use lookup_order. Do not "
            "use this for return shipments."
        ),
        input_model=OrderIdInput,
        tags=["orders", "shipping"],
        audit_log=audit,
    )
    def get_tracking_status(order_id: str) -> dict[str, Any]:
        order = ORDERS.get(order_id)
        if order is None:
            raise ToolException(ToolError(
                category=ErrorCategory.NOT_FOUND,
                message=f"No order with id {order_id!r} exists.",
                retryable=False,
            ))
        if order["status"] in ("pending", "processing"):
            raise ToolException(ToolError(
                category=ErrorCategory.PRECONDITION_FAILED,
                message="Order has not shipped yet — no tracking is available.",
                retryable=False,
                suggested_action=(
                    "Tell the customer the order is still being processed; "
                    "tracking will be available after it ships."
                ),
                details={"current_status": order["status"]},
            ))
        return tracking_payload(order)

    return get_tracking_status


def _make_initiate_return(audit: AuditLog, cache: IdempotencyCache) -> Tool:
    @tool(
        name="initiate_return",
        description=(
            "Start a return process for an order. Generates a return shipping "
            "label and return authorization number. Idempotent on order_id."
        ),
        when_not_to_use=(
            "Do not call if the order is older than 30 days — unless the customer "
            "reports a defect, in which case escalate to a human agent. Do not use "
            "for exchanges (not supported — process a return + new order). Do not "
            "use for cancellations — use cancel_order instead."
        ),
        input_model=ReturnInput,
        tags=["orders", "returns"],
        audit_log=audit,
        idempotent=True,
        idempotency_cache=cache,
        idempotency_key_fields=["order_id"],
    )
    def initiate_return(order_id: str, reason: str) -> dict[str, Any]:
        order = ORDERS.get(order_id)
        if order is None:
            raise ToolException(ToolError(
                category=ErrorCategory.NOT_FOUND,
                message=f"No order with id {order_id!r} exists.",
                retryable=False,
            ))
        age_days = days_since_order(order)
        if age_days > RETURN_WINDOW_DAYS:
            raise ToolException(ToolError(
                category=ErrorCategory.PRECONDITION_FAILED,
                message=(
                    f"Order is {age_days} days old; the return window is "
                    f"{RETURN_WINDOW_DAYS} days."
                ),
                retryable=False,
                suggested_action=(
                    "If the reason indicates a defect, escalate to a human agent. "
                    "Otherwise, explain the return-window policy."
                ),
                details={"age_days": age_days, "reason": reason},
            ))
        return {
            "return_id": f"RET-{uuid.uuid4().hex[:8].upper()}",
            "return_label_url": f"https://shopfast.example/returns/{order_id}.pdf",
            "refund_amount": order["total"],
            "refund_timeline": "5-7 business days after we receive the item",
        }

    return initiate_return


def _make_cancel_order(audit: AuditLog, cache: IdempotencyCache) -> Tool:
    @tool(
        name="cancel_order",
        description=(
            "Cancel an order that hasn't shipped yet. Immediately processes a "
            "full refund. Idempotent on order_id."
        ),
        when_not_to_use=(
            "Do not call for orders with status 'shipped' or 'delivered' — offer a "
            "return via initiate_return instead. Do not use for partial cancellations "
            "— only full order cancellation is supported."
        ),
        input_model=CancelInput,
        tags=["orders", "cancellations"],
        audit_log=audit,
        idempotent=True,
        idempotency_cache=cache,
        idempotency_key_fields=["order_id"],
    )
    def cancel_order(order_id: str, reason: str) -> dict[str, Any]:
        order = ORDERS.get(order_id)
        if order is None:
            raise ToolException(ToolError(
                category=ErrorCategory.NOT_FOUND,
                message=f"No order with id {order_id!r} exists.",
                retryable=False,
            ))
        if order["status"] in ("shipped", "delivered"):
            raise ToolException(ToolError(
                category=ErrorCategory.PRECONDITION_FAILED,
                message="Order has already shipped and cannot be cancelled.",
                retryable=False,
                suggested_action="Offer to initiate a return via initiate_return instead.",
                details={"current_status": order["status"], "reason": reason},
            ))
        return {
            "cancellation_id": f"CAN-{uuid.uuid4().hex[:8].upper()}",
            "refund_amount": order["total"],
            "refund_timeline": "3-5 business days",
        }

    return cancel_order


def _make_escalate(audit: AuditLog) -> Tool:
    @tool(
        name="escalate_to_human",
        description=(
            "Transfer the conversation to a human agent. Always provide a context "
            "summary so the human doesn't start from scratch."
        ),
        when_not_to_use=(
            "Do not escalate for routine requests you can handle (tracking, simple "
            "returns, cancellations). Do not escalate just because the customer is "
            "frustrated — acknowledge the frustration and try to solve the problem first."
        ),
        input_model=EscalateInput,
        tags=["escalation"],
        audit_log=audit,
    )
    def escalate_to_human(
        reason: str,
        priority: str,
        context_summary: str,
    ) -> dict[str, Any]:
        # priority is constrained to one of these three values by the Pydantic
        # model — no ``.get`` defaults needed, validation runs first.
        wait_by_priority = {"urgent": 2, "high": 8, "normal": 25}
        queue_by_priority = {"urgent": 3, "high": 7, "normal": 12}
        return {
            "transfer_id": f"TRN-{uuid.uuid4().hex[:8].upper()}",
            "estimated_wait_time": wait_by_priority[priority],
            "queue_position": queue_by_priority[priority],
            "accepted_reason": reason,
            "context_summary_received": context_summary,
        }

    return escalate_to_human


def _make_process_return_request(audit: AuditLog, cache: IdempotencyCache) -> Tool:
    @tool(
        name="process_return_request",
        description=(
            "End-to-end return workflow in a single call. Internally: looks up the "
            "order, checks the return-window eligibility, and initiates the return "
            "if eligible. If the order is past the window AND the reason indicates "
            "a defect, automatically escalates to a human with the assembled "
            "context. The agent calls one tool; the orchestration is deterministic.\n\n"
            "WHEN TO USE: Customer wants to return an order. Always prefer this "
            "over chaining lookup_order + initiate_return manually — the chain is "
            "fragile to step ordering and easy to forget the eligibility check."
        ),
        when_not_to_use=(
            "Do not use for cancellations of unshipped orders — use cancel_order. "
            "Do not use for tracking questions — use get_tracking_status."
        ),
        input_model=ReturnInput,
        tags=["orders", "returns", "fat"],
        audit_log=audit,
        idempotent=True,
        idempotency_cache=cache,
        idempotency_key_fields=["order_id"],
    )
    def process_return_request(order_id: str, reason: str) -> dict[str, Any]:
        # Step 1: lookup. NOT_FOUND is a hard stop — no escalation makes sense.
        order = ORDERS.get(order_id)
        if order is None:
            raise ToolException(ToolError(
                category=ErrorCategory.NOT_FOUND,
                message=f"No order with id {order_id!r} exists.",
                retryable=False,
                suggested_action=(
                    "Ask the customer to confirm the order number from their "
                    "confirmation email."
                ),
            ))

        # Step 2: eligibility check. This is the routing decision the LLM would
        # otherwise have to make — codified here so it can't be skipped.
        age_days = days_since_order(order)
        is_defect = any(
            term in reason.lower()
            for term in ("defect", "broken", "damaged", "faulty")
        )

        if age_days <= RETURN_WINDOW_DAYS:
            return {
                "outcome": "return_initiated",
                "order": {
                    "order_id": order["order_id"],
                    "items": order["items"],
                    "status": order["status"],
                    "order_date": order["order_date"],
                    "total": order["total"],
                },
                "return_id": f"RET-{uuid.uuid4().hex[:8].upper()}",
                "return_label_url": f"https://shopfast.example/returns/{order_id}.pdf",
                "refund_amount": order["total"],
                "refund_timeline": "5-7 business days after we receive the item",
            }

        if is_defect:
            return {
                "outcome": "escalated",
                "transfer_id": f"TRN-{uuid.uuid4().hex[:8].upper()}",
                "priority": "high",
                "estimated_wait_time": 8,
                "context_summary": (
                    f"Order {order_id} ({', '.join(order['items'])}) is "
                    f"{age_days} days old (past the {RETURN_WINDOW_DAYS}-day "
                    f"return window). Customer reports defect: {reason!r}. "
                    "Manual review needed for warranty/exception."
                ),
            }

        raise ToolException(ToolError(
            category=ErrorCategory.PRECONDITION_FAILED,
            message=(
                f"Order is {age_days} days old (return window is "
                f"{RETURN_WINDOW_DAYS} days) and the reason isn't a defect."
            ),
            retryable=False,
            suggested_action=(
                "Explain the return-window policy. If the customer reports a "
                "defect or warranty issue, retry the call with a clearer reason."
            ),
            details={"age_days": age_days, "reason": reason},
        ))

    return process_return_request


# -- Public surface -----------------------------------------------------------


def build_tools(
    audit_log: AuditLog | None = None,
    idempotency_cache: IdempotencyCache | None = None,
) -> tuple[list[Tool], AuditLog, IdempotencyCache]:
    """Construct fresh tools wired to the given audit log and idempotency cache.

    If you don't pass instances, fresh ones are created so two calls never
    share state. Returns the tools plus the resolved audit/cache so callers
    can read metrics or clear the cache without reaching into the closures.
    """
    audit = audit_log if audit_log is not None else AuditLog()
    cache = idempotency_cache if idempotency_cache is not None else IdempotencyCache()
    tools = [
        _make_lookup_order(audit),
        _make_get_tracking_status(audit),
        _make_initiate_return(audit, cache),
        _make_cancel_order(audit, cache),
        _make_escalate(audit),
        _make_process_return_request(audit, cache),
    ]
    return tools, audit, cache


def build_registry(
    audit_log: AuditLog | None = None,
    idempotency_cache: IdempotencyCache | None = None,
) -> CapabilityRegistry:
    """Construct a registry containing every e-commerce tool with fresh state."""
    tools, _, _ = build_tools(audit_log, idempotency_cache)
    return CapabilityRegistry(tools)


def get_handlers(
    audit_log: AuditLog | None = None,
    idempotency_cache: IdempotencyCache | None = None,
) -> dict[str, Tool]:
    """Return the handler map for ``AgentRunner.tool_handlers``.

    Drop-in compatible with agent-eval-loop's customer_support example —
    pass to ``ImprovementLoop(tool_handlers=...)``.
    """
    tools, _, _ = build_tools(audit_log, idempotency_cache)
    return {t.name: t for t in tools}
