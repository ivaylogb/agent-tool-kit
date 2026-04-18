"""Tests for the context-compression helpers."""

from __future__ import annotations

from agent_tool_kit import (
    highlight_fragments,
    project,
    project_many,
    truncate_text,
)


def test_project_keeps_only_named_fields():
    record = {"a": 1, "b": 2, "c": 3}
    assert project(record, ["a", "c"]) == {"a": 1, "c": 3}


def test_project_drops_missing_silently():
    assert project({"a": 1}, ["a", "missing"]) == {"a": 1}


def test_project_many_applies_per_record():
    records = [{"a": 1, "b": 2}, {"a": 10, "c": 30}]
    assert project_many(records, ["a"]) == [{"a": 1}, {"a": 10}]


def test_truncate_text_short_input_unchanged():
    assert truncate_text("hi", 10) == "hi"


def test_truncate_text_appends_suffix():
    out = truncate_text("hello world", 8)
    assert out.endswith("...")
    assert len(out) == 8


def test_truncate_text_zero_or_negative_max_returns_empty():
    assert truncate_text("hi", 0) == ""


def test_highlight_fragments_returns_spans_around_matches():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"
    fragments = highlight_fragments(text, ["gamma"], fragment_chars=20, max_fragments=2)
    assert len(fragments) == 1
    assert "gamma" in fragments[0]


def test_highlight_fragments_handles_multiple_terms():
    text = "alpha beta " + "x " * 100 + "gamma"
    fragments = highlight_fragments(text, ["alpha", "gamma"], fragment_chars=30, max_fragments=3)
    assert len(fragments) == 2
    assert any("alpha" in f for f in fragments)
    assert any("gamma" in f for f in fragments)


def test_highlight_fragments_empty_when_no_match():
    assert highlight_fragments("hello world", ["zzz"]) == []


def test_highlight_fragments_empty_inputs_safe():
    assert highlight_fragments("", ["x"]) == []
    assert highlight_fragments("hi", []) == []
    assert highlight_fragments("hi", [""]) == []


def test_highlight_fragments_max_fragments_respected():
    text = "match " * 50
    fragments = highlight_fragments(text, ["match"], fragment_chars=30, max_fragments=2)
    assert len(fragments) <= 2


def test_highlight_fragments_dedupes_overlapping_spans():
    # Adjacent matches should fold into a single emitted span.
    text = "match match match"
    fragments = highlight_fragments(text, ["match"], fragment_chars=200, max_fragments=5)
    assert len(fragments) == 1
