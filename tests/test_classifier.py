"""Tests for the lightweight tool selection classifier.

The Anthropic client is stubbed so the test suite runs offline.
"""

from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from agent_tool_kit import CapabilityRegistry, ToolClassifier, tool


class _Inp(BaseModel):
    x: str


def make_registry():
    @tool(name="lookup_order", description="Look up order.", input_model=_Inp)
    def lookup_order(x: str) -> dict:
        return {}

    @tool(name="search_kb", description="Search knowledge base.", input_model=_Inp)
    def search_kb(x: str) -> dict:
        return {}

    @tool(name="cancel_order", description="Cancel order.", input_model=_Inp)
    def cancel_order(x: str) -> dict:
        return {}

    return CapabilityRegistry([lookup_order, search_kb, cancel_order])


class FakeAnthropicClient:
    """Stub that returns a canned text response shaped like Messages API."""

    def __init__(self, text: str):
        self._text = text
        self.messages = self
        self.last_call: dict | None = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._text)]
        )


def test_classifier_returns_selected_subset():
    reg = make_registry()
    client = FakeAnthropicClient('{"selected": ["lookup_order", "cancel_order"], "reasoning": "x"}')
    classifier = ToolClassifier(reg, client=client)  # type: ignore[arg-type]
    assert classifier.select("Cancel ORD-1") == ["lookup_order", "cancel_order"]


def test_classifier_filters_unknown_names():
    reg = make_registry()
    client = FakeAnthropicClient('{"selected": ["lookup_order", "ghost_tool"]}')
    classifier = ToolClassifier(reg, client=client)  # type: ignore[arg-type]
    assert classifier.select("x") == ["lookup_order"]


def test_classifier_falls_back_when_response_unparseable():
    reg = make_registry()
    client = FakeAnthropicClient("this is not json at all")
    classifier = ToolClassifier(reg, client=client)  # type: ignore[arg-type]
    # Falls back to the full menu so the executor isn't blocked by a bad classifier.
    assert classifier.select("x") == reg.names()


def test_classifier_fallback_capped_by_max_tools():
    reg = make_registry()
    client = FakeAnthropicClient("garbage")
    classifier = ToolClassifier(reg, client=client)  # type: ignore[arg-type]
    assert classifier.select("x", max_tools=2) == reg.names()[:2]


def test_classifier_returns_empty_for_empty_registry():
    reg = CapabilityRegistry()
    classifier = ToolClassifier(reg, client=FakeAnthropicClient("{}"))  # type: ignore[arg-type]
    assert classifier.select("x") == []


def test_classifier_passes_menu_to_system_prompt():
    reg = make_registry()
    client = FakeAnthropicClient('{"selected": ["lookup_order"]}')
    classifier = ToolClassifier(reg, client=client)  # type: ignore[arg-type]
    classifier.select("Look up order ORD-1")
    assert client.last_call is not None
    sys_prompt = client.last_call["system"]
    for name in reg.names():
        assert name in sys_prompt
