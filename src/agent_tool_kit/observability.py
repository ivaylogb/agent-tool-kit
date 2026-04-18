"""Audit log + metrics for tool calls.

Tools are the agent's interface to the real world. When a tool is misused,
the agent produces incorrect responses even if the prompt is perfect. The audit
trail is the primary debugging signal: every call, with inputs, outputs,
latency, and errors, written to a JSONL file the engineer can replay.

The log is append-only and durable per record — each call flushes before the
handler returns, so a crash mid-conversation still leaves a complete trail.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from agent_tool_kit.errors import ToolError


class ToolCallRecord(BaseModel):
    """One audit log entry per tool invocation."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    error: dict[str, Any] | None = None
    idempotent_hit: bool = False
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str | None = None


class AuditLog:
    """Append-only audit log for tool invocations.

    Records are kept in memory and (if ``path`` is set) flushed to a JSONL file.
    Use ``replay`` to iterate historical entries — useful for reproducing
    failures without re-running the LLM.

    Latency is measured from ``begin(call_id)`` to ``record(call_id=...)``.
    The ``Tool`` wrapper handles both calls; manual users can also call them
    directly for handlers built outside the toolkit.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        session_id: str | None = None,
    ):
        self.path = Path(path) if path is not None else None
        self.session_id = session_id
        self._records: list[ToolCallRecord] = []
        self._start_times: dict[str, float] = {}

    def begin(self, call_id: str) -> None:
        """Mark the start time of a call so ``record`` can compute latency."""
        self._start_times[call_id] = time.time()

    def record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: ToolError | None = None,
        idempotent_hit: bool = False,
        call_id: str | None = None,
    ) -> ToolCallRecord:
        cid = call_id or uuid.uuid4().hex
        start = self._start_times.pop(cid, None)
        latency_ms = (time.time() - start) * 1000.0 if start is not None else 0.0
        record = ToolCallRecord(
            call_id=cid,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            error=error.model_dump(mode="json") if error is not None else None,
            idempotent_hit=idempotent_hit,
            latency_ms=latency_ms,
            session_id=self.session_id,
        )
        self._records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(record.model_dump_json() + "\n")
        return record

    def records(self) -> list[ToolCallRecord]:
        return list(self._records)

    def metrics(self) -> dict[str, Any]:
        """Aggregate stats across recorded calls. Per-tool breakdown included."""
        if not self._records:
            return {
                "calls": 0,
                "errors": 0,
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
                "per_tool": {},
            }
        per_tool: dict[str, dict[str, float]] = {}
        for r in self._records:
            d = per_tool.setdefault(
                r.tool_name,
                {"calls": 0.0, "errors": 0.0, "_latency_total": 0.0},
            )
            d["calls"] += 1
            if r.error is not None:
                d["errors"] += 1
            d["_latency_total"] += r.latency_ms
        for d in per_tool.values():
            d["avg_latency_ms"] = d["_latency_total"] / d["calls"] if d["calls"] else 0.0
            d.pop("_latency_total")
        total_errors = sum(int(d["errors"]) for d in per_tool.values())
        total_latency = sum(r.latency_ms for r in self._records)
        return {
            "calls": len(self._records),
            "errors": total_errors,
            "error_rate": total_errors / len(self._records),
            "avg_latency_ms": total_latency / len(self._records),
            "per_tool": per_tool,
        }

    @classmethod
    def replay(cls, path: str | Path) -> Iterator[ToolCallRecord]:
        """Iterate records from a previously written JSONL audit log."""
        p = Path(path)
        if not p.exists():
            return
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield ToolCallRecord.model_validate_json(line)
