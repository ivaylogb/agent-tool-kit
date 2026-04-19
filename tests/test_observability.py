"""Tests for the audit log + replay + metrics."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from agent_tool_kit import AuditLog, ErrorCategory, ToolError, tool


class Inp(BaseModel):
    x: int


def test_record_writes_to_jsonl(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path, session_id="s1")

    @tool(name="bump", description="Bump.", input_model=Inp, audit_log=log)
    def bump(x: int) -> dict:
        return {"x": x + 1}

    bump(x=1)
    bump(x=2)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["tool_name"] == "bump"
    assert rec["arguments"] == {"x": 1}
    assert rec["result"] == {"x": 2}
    assert rec["session_id"] == "s1"
    assert "timestamp" in rec
    assert rec["latency_ms"] >= 0


def test_replay_round_trips(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path)

    @tool(name="bump", description="Bump.", input_model=Inp, audit_log=log)
    def bump(x: int) -> dict:
        return {"x": x + 1}

    bump(x=1)
    bump(x=2)

    replayed = list(AuditLog.replay(path))
    assert len(replayed) == 2
    assert replayed[0].arguments == {"x": 1}
    assert replayed[1].arguments == {"x": 2}


def test_replay_missing_file_yields_nothing(tmp_path: Path):
    assert list(AuditLog.replay(tmp_path / "nope.jsonl")) == []


def test_metrics_aggregates_per_tool():
    log = AuditLog()

    @tool(name="ok_tool", description="x.", input_model=Inp, audit_log=log)
    def ok_tool(x: int) -> dict:
        return {"ok": True}

    @tool(name="err_tool", description="x.", input_model=Inp, audit_log=log)
    def err_tool(x: int) -> dict:
        raise RuntimeError("boom")

    ok_tool(x=1)
    ok_tool(x=2)
    err_tool(x=3)

    m = log.metrics()
    assert m["calls"] == 3
    assert m["errors"] == 1
    assert m["error_rate"] > 0
    assert "ok_tool" in m["per_tool"]
    assert m["per_tool"]["ok_tool"]["calls"] == 2
    assert m["per_tool"]["err_tool"]["errors"] == 1


def test_metrics_empty_log_safe():
    log = AuditLog()
    m = log.metrics()
    assert m == {
        "calls": 0,
        "errors": 0,
        "error_rate": 0.0,
        "avg_latency_ms": 0.0,
        "per_tool": {},
    }


def test_record_serializes_tool_error_in_envelope():
    log = AuditLog()
    log.begin("c1")
    err = ToolError(category=ErrorCategory.NOT_FOUND, message="x")
    rec = log.record(
        tool_name="t",
        arguments={"a": 1},
        result=err.to_response(),
        error=err,
        call_id="c1",
    )
    assert rec.error == err.model_dump(mode="json")


def test_audit_log_is_thread_safe(tmp_path: Path):
    """Concurrent producers must not corrupt _records or interleave JSONL lines."""
    import threading

    path = tmp_path / "concurrent.jsonl"
    log = AuditLog(path=path)

    @tool(name="bump", description="Bump.", input_model=Inp, audit_log=log)
    def bump(x: int) -> dict:
        return {"x": x}

    n_threads = 8
    calls_per_thread = 50

    def worker(start: int) -> None:
        for i in range(start, start + calls_per_thread):
            bump(x=i)

    threads = [
        threading.Thread(target=worker, args=(t * calls_per_thread,))
        for t in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = n_threads * calls_per_thread

    # In-memory records all present.
    assert len(log.records()) == expected

    # JSONL on disk has one well-formed record per line.
    with path.open() as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == expected
    for line in lines:
        # Each line parses as a complete JSON object — no interleaving.
        parsed = json.loads(line)
        assert parsed["tool_name"] == "bump"


def test_discard_pending_clears_orphan_timer():
    log = AuditLog()
    log.begin("c-orphan")
    # Caller never produced a record; clean up explicitly.
    log.discard_pending("c-orphan")
    # Recording with the same id now reports zero latency — proving the
    # orphan was actually removed (otherwise latency would be huge).
    rec = log.record(tool_name="t", arguments={}, call_id="c-orphan")
    assert rec.latency_ms == 0.0


def test_metrics_per_tool_counts_are_ints_not_floats():
    """Cosmetic regression check: per-tool counts must serialize as ints."""
    log = AuditLog()

    @tool(name="t", description="x.", input_model=Inp, audit_log=log)
    def t(x: int) -> dict:
        return {"x": x}

    t(x=1)
    t(x=2)
    m = log.metrics()
    assert m["per_tool"]["t"]["calls"] == 2
    assert isinstance(m["per_tool"]["t"]["calls"], int)
    assert isinstance(m["per_tool"]["t"]["errors"], int)
