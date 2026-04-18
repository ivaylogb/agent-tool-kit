"""Tests for the idempotency cache."""

from __future__ import annotations

from agent_tool_kit import IdempotencyCache


def test_get_returns_none_when_missing():
    cache = IdempotencyCache()
    assert cache.get("tool", "key") is None


def test_put_then_get():
    cache = IdempotencyCache()
    cache.put("tool", "key", {"x": 1})
    assert cache.get("tool", "key") == {"x": 1}


def test_keys_scoped_per_tool():
    cache = IdempotencyCache()
    cache.put("a", "key", "value-a")
    cache.put("b", "key", "value-b")
    assert cache.get("a", "key") == "value-a"
    assert cache.get("b", "key") == "value-b"


def test_clear_empties_cache():
    cache = IdempotencyCache()
    cache.put("a", "k", 1)
    cache.put("b", "k", 2)
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a", "k") is None
