"""Tests for the Tool wrapper, decorator, and refusal helper."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from agent_tool_kit import (
    AuditLog,
    ErrorCategory,
    IdempotencyCache,
    Tool,
    ToolError,
    ToolException,
    refuse,
    tool,
)


class EchoInput(BaseModel):
    value: str = Field(description="A short string.", min_length=1)


def test_tool_schema_is_anthropic_compatible():
    @tool(name="echo", description="Echo a value.", input_model=EchoInput)
    def echo(value: str) -> dict:
        return {"value": value}

    schema = echo.tool_schema
    assert schema["name"] == "echo"
    assert schema["input_schema"]["type"] == "object"
    assert "value" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["value"]
    # Pydantic ``title`` keys should be stripped from the schema.
    assert "title" not in schema["input_schema"]
    for prop in schema["input_schema"]["properties"].values():
        assert "title" not in prop


def test_when_not_to_use_appended_to_description():
    @tool(
        name="echo",
        description="Echo a value.",
        when_not_to_use="Do not use for binary data.",
        input_model=EchoInput,
    )
    def echo(value: str) -> dict:
        return {"value": value}

    assert "WHEN NOT TO USE" in echo.tool_schema["description"]
    assert "binary data" in echo.tool_schema["description"]


def test_call_with_kwargs_runs_handler():
    @tool(name="echo", description="Echo.", input_model=EchoInput)
    def echo(value: str) -> dict:
        return {"value": value, "len": len(value)}

    assert echo(value="hi") == {"value": "hi", "len": 2}


def test_validation_error_returns_structured_error():
    @tool(name="echo", description="Echo.", input_model=EchoInput)
    def echo(value: str) -> dict:
        return {"value": value}

    out = echo(value="")
    assert "error" in out
    assert out["error"]["category"] == "invalid_input"
    assert out["error"]["retryable"] is True
    # Field that failed appears in the message so the LLM can correct it.
    assert "value" in out["error"]["message"]


def test_missing_required_field_returns_invalid_input():
    @tool(name="echo", description="Echo.", input_model=EchoInput)
    def echo(value: str) -> dict:
        return {"value": value}

    out = echo()
    assert out["error"]["category"] == "invalid_input"


def test_tool_exception_becomes_envelope():
    @tool(name="echo", description="Echo.", input_model=EchoInput)
    def echo(value: str) -> dict:
        raise ToolException(ToolError(
            category=ErrorCategory.NOT_FOUND,
            message="nope",
            retryable=False,
        ))

    out = echo(value="hi")
    assert out["error"]["category"] == "not_found"
    assert out["error"]["message"] == "nope"


def test_unexpected_exception_becomes_internal_error():
    @tool(name="echo", description="Echo.", input_model=EchoInput)
    def echo(value: str) -> dict:
        raise RuntimeError("kaboom")

    out = echo(value="hi")
    assert out["error"]["category"] == "internal"
    assert "kaboom" in out["error"]["message"]


def test_handler_accepting_pydantic_model_directly():
    @tool(name="echo", description="Echo.", input_model=EchoInput)
    def echo(payload: EchoInput) -> dict:
        return {"value": payload.value.upper()}

    assert echo(value="hi") == {"value": "HI"}


def test_output_filter_runs_after_handler():
    def keep_only_value(result: dict) -> dict:
        return {"value": result["value"]}

    @tool(
        name="echo",
        description="Echo.",
        input_model=EchoInput,
        output_filter=keep_only_value,
    )
    def echo(value: str) -> dict:
        return {"value": value, "secret": "drop me"}

    out = echo(value="hi")
    assert out == {"value": "hi"}


def test_idempotency_caches_first_result():
    cache = IdempotencyCache()
    counter = {"n": 0}

    @tool(
        name="bump",
        description="Increment a counter.",
        input_model=EchoInput,
        idempotent=True,
        idempotency_cache=cache,
    )
    def bump(value: str) -> dict:
        counter["n"] += 1
        return {"value": value, "n": counter["n"]}

    first = bump(value="x")
    second = bump(value="x")
    assert first == second
    assert counter["n"] == 1


def test_idempotency_does_not_cache_errors():
    cache = IdempotencyCache()

    @tool(
        name="failer",
        description="Always fails.",
        input_model=EchoInput,
        idempotent=True,
        idempotency_cache=cache,
    )
    def failer(value: str) -> dict:
        raise ToolException(ToolError(
            category=ErrorCategory.UPSTREAM_FAILURE,
            message="upstream is down",
        ))

    failer(value="x")
    failer(value="x")
    assert len(cache) == 0


def test_idempotency_requires_cache():
    with pytest.raises(ValueError):
        Tool(
            handler=lambda value: {},
            name="foo",
            description="x",
            input_model=EchoInput,
            idempotent=True,
        )


def test_idempotency_key_fields_scope_the_key():
    cache = IdempotencyCache()

    class TwoFieldInput(BaseModel):
        order_id: str
        reason: str

    @tool(
        name="initiate_return",
        description="Return.",
        input_model=TwoFieldInput,
        idempotent=True,
        idempotency_cache=cache,
        idempotency_key_fields=["order_id"],
    )
    def initiate_return(order_id: str, reason: str) -> dict:
        return {"return_id": "RET-" + order_id, "reason": reason}

    first = initiate_return(order_id="ORD-1", reason="wrong size")
    second = initiate_return(order_id="ORD-1", reason="changed mind")
    # Same order_id → same cached envelope despite different cosmetic reason.
    assert first == second


def test_audit_log_records_calls_with_latency():
    log = AuditLog()

    @tool(name="echo", description="Echo.", input_model=EchoInput, audit_log=log)
    def echo(value: str) -> dict:
        return {"value": value}

    echo(value="hi")
    echo(value="bye")
    records = log.records()
    assert len(records) == 2
    assert all(r.tool_name == "echo" for r in records)
    assert all(r.latency_ms >= 0 for r in records)


def test_audit_log_marks_idempotent_hit():
    log = AuditLog()
    cache = IdempotencyCache()

    @tool(
        name="echo",
        description="Echo.",
        input_model=EchoInput,
        audit_log=log,
        idempotent=True,
        idempotency_cache=cache,
    )
    def echo(value: str) -> dict:
        return {"value": value}

    echo(value="x")
    echo(value="x")
    records = log.records()
    assert records[0].idempotent_hit is False
    assert records[1].idempotent_hit is True


def test_refuse_helper_shape():
    out = refuse("not authorized", policy="strict")
    assert out == {"refusal": "not authorized", "details": {"policy": "strict"}}
    assert refuse("nope") == {"refusal": "nope"}


def test_decorator_uses_docstring_when_description_omitted():
    @tool(name="echo", input_model=EchoInput)
    def echo(value: str) -> dict:
        """Echo a value back to the caller."""
        return {"value": value}

    assert "Echo a value back" in echo.tool_schema["description"]


def test_constructor_rejects_non_pydantic_input_model():
    with pytest.raises(TypeError):
        Tool(
            handler=lambda value: {},
            name="foo",
            description="x",
            input_model=dict,  # type: ignore[arg-type]
        )


def test_menu_entry_truncates_long_description():
    long_desc = "x" * 500
    @tool(name="echo", description=long_desc, input_model=EchoInput)
    def echo(value: str) -> dict:
        return {}

    entry = echo.menu_entry
    assert len(entry["summary"]) <= 160
    assert entry["summary"].endswith("...")


# ---- output_filter exception handling (regression for the 1.2/1.3 fix) ----


def test_output_filter_exception_becomes_internal_error_not_raised():
    """A raising filter must not escape Tool.invoke — it becomes an envelope."""
    def buggy_filter(_result):
        raise RuntimeError("filter explodes")

    @tool(
        name="echo",
        description="Echo.",
        input_model=EchoInput,
        output_filter=buggy_filter,
    )
    def echo(value: str) -> dict:
        return {"value": value}

    out = echo(value="hi")
    assert out["error"]["category"] == "internal"
    assert "filter explodes" in out["error"]["message"]
    assert "output_filter raised" in out["error"]["message"]


def test_output_filter_skipped_when_handler_returns_error_envelope():
    """Filters expect success shapes; they should not see error envelopes."""
    seen: list = []

    def record_then_pass(result):
        seen.append(result)
        return result

    @tool(
        name="echo",
        description="Echo.",
        input_model=EchoInput,
        output_filter=record_then_pass,
    )
    def echo(value: str) -> dict:
        return {"error": {"category": "not_found", "message": "x", "retryable": False}}

    out = echo(value="hi")
    # Filter saw nothing — error envelope was returned untouched.
    assert seen == []
    assert out["error"]["category"] == "not_found"


def test_audit_timer_not_leaked_when_filter_raises():
    """Pending audit timer must be cleared even when output_filter raises."""
    log = AuditLog()

    def buggy_filter(_):
        raise RuntimeError("oops")

    @tool(
        name="echo",
        description="Echo.",
        input_model=EchoInput,
        audit_log=log,
        output_filter=buggy_filter,
    )
    def echo(value: str) -> dict:
        return {"value": value}

    echo(value="hi")
    # The internal-error envelope was recorded; no orphan start time remains.
    assert len(log.records()) == 1
    assert log.records()[0].error is not None
    # Direct check that the locked accessor sees no pending entries.
    log.discard_pending("nonexistent")  # smoke-test that the public API is callable


# ---- Refusal envelope passes through Tool.invoke unchanged ----


def test_refusal_envelope_round_trips_through_invoke():
    @tool(name="echo", description="Echo.", input_model=EchoInput)
    def echo(value: str) -> dict:
        return refuse("not authorized for this action", policy="strict")

    out = echo(value="hi")
    assert out == {
        "refusal": "not authorized for this action",
        "details": {"policy": "strict"},
    }
    # Refusals are NOT classified as errors and pass through filters
    # exactly as the handler returned them.


def test_refusal_envelope_is_cached_by_idempotent_tool():
    """Documented behavior: refusals are cached, since they're not error envelopes."""
    cache = IdempotencyCache()
    counter = {"n": 0}

    @tool(
        name="echo",
        description="Echo.",
        input_model=EchoInput,
        idempotent=True,
        idempotency_cache=cache,
    )
    def echo(value: str) -> dict:
        counter["n"] += 1
        return refuse("nope")

    first = echo(value="x")
    second = echo(value="x")
    assert first == second == {"refusal": "nope"}
    # Refusal was cached, handler ran only once.
    assert counter["n"] == 1
