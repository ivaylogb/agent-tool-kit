"""Tests for the CapabilityRegistry."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from agent_tool_kit import CapabilityRegistry, tool


class EchoInput(BaseModel):
    value: str = Field(description="A string.")


def make_tool(name: str, tags: list[str] | None = None):
    @tool(
        name=name,
        description=f"Tool {name}.",
        input_model=EchoInput,
        tags=tags,
    )
    def fn(value: str) -> dict:
        return {"value": value, "tool": name}

    return fn


def test_register_and_lookup():
    t = make_tool("a")
    reg = CapabilityRegistry([t])
    assert "a" in reg
    assert reg.get("a") is t
    assert len(reg) == 1
    assert reg.names() == ["a"]


def test_duplicate_registration_raises():
    t = make_tool("a")
    reg = CapabilityRegistry([t])
    with pytest.raises(ValueError):
        reg.register(make_tool("a"))


def test_menu_returns_lightweight_entries():
    reg = CapabilityRegistry([
        make_tool("a", tags=["x"]),
        make_tool("b", tags=["y"]),
    ])
    menu = reg.menu()
    assert len(menu) == 2
    assert {m["name"] for m in menu} == {"a", "b"}
    assert all("description" in m for m in menu)
    assert all("tags" in m for m in menu)


def test_menu_filtered_by_tag():
    reg = CapabilityRegistry([
        make_tool("a", tags=["x"]),
        make_tool("b", tags=["y"]),
        make_tool("c", tags=["x", "z"]),
    ])
    names = {m["name"] for m in reg.menu(tags=["x"])}
    assert names == {"a", "c"}


def test_schemas_for_named_subset():
    reg = CapabilityRegistry([make_tool("a"), make_tool("b"), make_tool("c")])
    schemas = reg.schemas_for(["a", "c"])
    assert [s["name"] for s in schemas] == ["a", "c"]


def test_schemas_for_unknown_raises():
    reg = CapabilityRegistry([make_tool("a")])
    with pytest.raises(KeyError):
        reg.schemas_for(["a", "missing"])


def test_handlers_returns_callable_map():
    t = make_tool("a")
    reg = CapabilityRegistry([t])
    handlers = reg.handlers()
    assert handlers["a"] is t
    assert handlers["a"](value="hi") == {"value": "hi", "tool": "a"}


def test_handlers_subset_skips_unknown():
    reg = CapabilityRegistry([make_tool("a"), make_tool("b")])
    handlers = reg.handlers(["a", "missing"])
    assert set(handlers) == {"a"}


def test_execute_unknown_returns_structured_error():
    reg = CapabilityRegistry([make_tool("a")])
    out = reg.execute("nope", {})
    assert out["error"]["category"] == "not_found"
    assert "Available: a" in out["error"]["message"]


def test_execute_unknown_with_empty_registry():
    reg = CapabilityRegistry()
    out = reg.execute("nope", {})
    assert out["error"]["category"] == "not_found"
    assert "<empty>" in out["error"]["message"]


def test_iter_yields_tools():
    a, b = make_tool("a"), make_tool("b")
    reg = CapabilityRegistry([a, b])
    assert set(iter(reg)) == {a, b}
