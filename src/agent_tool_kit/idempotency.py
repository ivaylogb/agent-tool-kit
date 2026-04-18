"""Idempotency cache for safely re-callable action tools.

Why this exists: agents retry. Context windows truncate. Models occasionally
emit a duplicate ``tool_use`` block. If ``create_return`` runs twice because
the agent didn't see the first response, you don't want two return labels.

The cache stores the first successful result keyed by ``(tool_name,
derived_key)``. Subsequent calls with the same key return the cached
envelope. The Tool wrapper handles key derivation; this module just stores.

For multi-process deployments, replace ``IdempotencyCache`` with a Redis-backed
implementation that exposes the same ``get``/``put``/``clear`` interface.
"""

from __future__ import annotations

from threading import Lock
from typing import Any


class IdempotencyCache:
    """Thread-safe in-memory idempotency cache.

    Keys are scoped per tool to prevent collisions between tools that happen
    to compute the same input hash. The cache is unbounded — wrap or replace
    if memory pressure is a concern in your deployment.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], Any] = {}
        self._lock = Lock()

    def get(self, tool_name: str, key: str) -> Any | None:
        with self._lock:
            return self._cache.get((tool_name, key))

    def put(self, tool_name: str, key: str, value: Any) -> None:
        with self._lock:
            self._cache[(tool_name, key)] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: tuple[str, str]) -> bool:
        with self._lock:
            return key in self._cache
