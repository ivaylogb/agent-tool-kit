"""Smoke tests for the worked examples — every example must actually run."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from examples.ecommerce.tools import build_registry as build_ecommerce_registry  # noqa: E402
from examples.external_api.fake_api import FakeWeatherAPI  # noqa: E402
from examples.external_api.tools import build_weather_tool  # noqa: E402
from examples.knowledge_base.tools import build_registry as build_kb_registry  # noqa: E402

# ---------------------------------------------------------------- e-commerce


def test_ecommerce_lookup_success():
    reg = build_ecommerce_registry()
    out = reg.execute("lookup_order", {"order_id": "ORD-78234"})
    assert out["order_id"] == "ORD-78234"
    assert out["status"] == "shipped"


def test_ecommerce_lookup_invalid_format_returns_invalid_input():
    reg = build_ecommerce_registry()
    out = reg.execute("lookup_order", {"order_id": "garbage"})
    assert out["error"]["category"] == "invalid_input"


def test_ecommerce_lookup_unknown_order_returns_not_found():
    reg = build_ecommerce_registry()
    out = reg.execute("lookup_order", {"order_id": "ORD-99999"})
    assert out["error"]["category"] == "not_found"


def test_ecommerce_get_tracking_for_processing_order_fails_precondition():
    reg = build_ecommerce_registry()
    out = reg.execute("get_tracking_status", {"order_id": "ORD-91001"})
    assert out["error"]["category"] == "precondition_failed"


def test_ecommerce_cancel_for_shipped_order_fails_precondition():
    reg = build_ecommerce_registry()
    out = reg.execute("cancel_order", {"order_id": "ORD-78234", "reason": "ordered by mistake"})
    assert out["error"]["category"] == "precondition_failed"


def test_ecommerce_fat_tool_eligible_path():
    reg = build_ecommerce_registry()
    out = reg.execute(
        "process_return_request",
        {"order_id": "ORD-78234", "reason": "wrong size"},
    )
    assert out["outcome"] == "return_initiated"
    assert "return_id" in out
    assert out["refund_amount"] == 129.99


def test_ecommerce_fat_tool_defect_outside_window_escalates():
    reg = build_ecommerce_registry()
    out = reg.execute(
        "process_return_request",
        {"order_id": "ORD-20100", "reason": "broken on arrival"},
    )
    assert out["outcome"] == "escalated"
    assert "transfer_id" in out


def test_ecommerce_fat_tool_outside_window_no_defect_refused():
    reg = build_ecommerce_registry()
    out = reg.execute(
        "process_return_request",
        {"order_id": "ORD-20100", "reason": "changed my mind"},
    )
    assert out["error"]["category"] == "precondition_failed"


# ---------------------------------------------------------------- knowledge


def test_kb_search_returns_compressed_results():
    reg = build_kb_registry()
    out = reg.execute(
        "search_knowledge_base",
        {
            "query": "How long do I have to return defective items?",
            "top_k": 3,
            "fragment_chars": 200,
        },
    )
    assert "results" in out
    assert len(out["results"]) >= 1
    first = out["results"][0]
    assert set(first) == {"doc_id", "title", "tags", "score", "fragments"}
    # Bodies are NEVER returned by search.
    assert "body" not in first


def test_kb_search_stopwords_only_returns_invalid_input():
    reg = build_kb_registry()
    out = reg.execute(
        "search_knowledge_base",
        {"query": "the and or", "top_k": 3, "fragment_chars": 200},
    )
    assert out["error"]["category"] == "invalid_input"


def test_kb_fetch_document_returns_documented_fields_only():
    reg = build_kb_registry()
    out = reg.execute("fetch_document", {"doc_id": "kb-001", "max_chars": 2000})
    assert set(out) == {"doc_id", "title", "tags", "body", "truncated", "full_length"}


def test_kb_fetch_document_invalid_format_returns_invalid_input():
    reg = build_kb_registry()
    out = reg.execute("fetch_document", {"doc_id": "not-an-id", "max_chars": 1000})
    assert out["error"]["category"] == "invalid_input"


def test_kb_fetch_unknown_doc_returns_not_found():
    reg = build_kb_registry()
    out = reg.execute("fetch_document", {"doc_id": "kb-999", "max_chars": 1000})
    assert out["error"]["category"] == "not_found"


def test_kb_search_truncation_at_top_k():
    reg = build_kb_registry()
    out = reg.execute(
        "search_knowledge_base",
        {
            "query": "policy returns shipping warranty cancellation account",
            "top_k": 2,
            "fragment_chars": 200,
        },
    )
    assert len(out["results"]) <= 2


# ---------------------------------------------------------------- external


def test_external_api_retries_recover_transient_failures():
    api = FakeWeatherAPI(failure_pattern=["error", "error", "ok"])
    tool, deps = build_weather_tool(api=api, timeout_seconds=1.0)
    out = tool(city="Berlin")
    assert out["city"] == "Berlin"
    assert api.call_count == 3


def test_external_api_idempotency_skips_upstream_on_dup():
    api = FakeWeatherAPI(failure_pattern=["ok"])
    tool, _ = build_weather_tool(api=api, timeout_seconds=1.0)
    tool(city="Berlin")
    upstream_after_first = api.call_count
    tool(city="Berlin")
    assert api.call_count == upstream_after_first


def test_external_api_timeout_returns_structured_error():
    api = FakeWeatherAPI(failure_pattern=["timeout"], slow_seconds=2.0)
    tool, _ = build_weather_tool(api=api, timeout_seconds=0.1)
    out = tool(city="Tokyo")
    assert out["error"]["category"] == "timeout"
    assert out["error"]["retryable"] is True


def test_external_api_circuit_breaker_trips_after_threshold():
    from agent_tool_kit import CircuitBreaker, RetryPolicy

    api = FakeWeatherAPI(failure_pattern=["error"])
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0)
    tool, deps = build_weather_tool(
        api=api,
        timeout_seconds=1.0,
        breaker=breaker,
        retry_policy=RetryPolicy(max_attempts=1, base_delay=0, jitter=0),
    )
    tool(city="A")
    tool(city="B")
    out = tool(city="C")
    assert out["error"]["category"] == "circuit_open"


def test_external_api_projects_only_documented_fields():
    api = FakeWeatherAPI(failure_pattern=["ok"])
    tool, _ = build_weather_tool(api=api)
    out = tool(city="Berlin")
    expected = {"city", "temperature_c", "conditions", "humidity_pct", "wind_kmh", "observed_at"}
    assert set(out) == expected
