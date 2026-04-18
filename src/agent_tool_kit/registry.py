"""Capability registry with progressive disclosure.

Why this exists: at scale you can't load all tool descriptions into the context
window. With 20+ tools the agent's selection accuracy degrades — the
"10% hit rate problem". The registry separates two concerns:

- **Menu** (Level 1, always loaded): name + one-line summary + tags.
  ~50-100 tokens per tool. The agent — or a routing classifier — sees the
  catalogue without paying for every full schema.
- **Schemas** (Level 2, loaded on demand): full Anthropic-format tool
  definitions. Loaded only for the subset relevant to the current task.

Pair the registry with ``ToolClassifier`` to do the selection step with a
cheap model (Haiku), then load only the selected schemas into the executor's
context. The expensive model focuses on the actual work.
"""

from __future__ import annotations

from typing import Any, Iterable

from agent_tool_kit.base import Tool
from agent_tool_kit.errors import ErrorCategory, ToolError


class CapabilityRegistry:
    """A registry of tools with two-level disclosure.

    Tools register once and can then be retrieved either as menu entries
    (cheap) or as full Anthropic-format schemas (full). Use ``handlers``
    to get the callable map for ``AgentRunner.tool_handlers``.
    """

    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    # ----------------------------------------------------------------- mutate

    def register(self, tool_obj: Tool) -> Tool:
        """Add a tool. Re-registration of an existing name raises ValueError."""
        if tool_obj.name in self._tools:
            raise ValueError(f"Tool {tool_obj.name!r} is already registered.")
        self._tools[tool_obj.name] = tool_obj
        return tool_obj

    # ----------------------------------------------------------------- query

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    # ------------------------------------------------------- progressive disclosure

    def menu(self, tags: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Lightweight metadata catalogue. Always cheap to load.

        ``tags`` filters the menu down to tools that match at least one tag.
        Use this to scope the catalogue per workflow (e.g., only
        ``orders``-tagged tools for an order-tracking conversation).
        """
        tag_set = set(tags) if tags else None
        out: list[dict[str, Any]] = []
        for t in self._tools.values():
            if tag_set is not None and not (set(t.tags) & tag_set):
                continue
            out.append(t.menu_entry)
        return out

    def schemas_for(self, names: Iterable[str]) -> list[dict[str, Any]]:
        """Full Anthropic-API schemas for the named subset.

        Raises KeyError on unknown names — failing loudly is better than
        silently advertising a tool the registry can't execute.
        """
        out: list[dict[str, Any]] = []
        for name in names:
            t = self._tools.get(name)
            if t is None:
                raise KeyError(f"Unknown tool: {name!r}")
            out.append(t.tool_schema)
        return out

    def all_schemas(self) -> list[dict[str, Any]]:
        return [t.tool_schema for t in self._tools.values()]

    def handlers(self, names: Iterable[str] | None = None) -> dict[str, Tool]:
        """Map of name → callable for AgentRunner.tool_handlers.

        Pass ``names`` to scope to a subset (e.g., the classifier's selection).
        Without it, all registered tools are returned.
        """
        if names is None:
            return dict(self._tools)
        out: dict[str, Tool] = {}
        for n in names:
            t = self._tools.get(n)
            if t is not None:
                out[n] = t
        return out

    # ------------------------------------------------------------------ exec

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a registered tool by name. Unknown tools return a structured error."""
        t = self._tools.get(name)
        if t is None:
            available = ", ".join(sorted(self._tools.keys())) or "<empty>"
            return ToolError(
                category=ErrorCategory.NOT_FOUND,
                message=f"Tool {name!r} is not registered. Available: {available}",
                retryable=False,
                suggested_action=(
                    "Re-issue the call with one of the available tool names "
                    "above, or stop and ask the user for clarification."
                ),
                details={"available_tools": sorted(self._tools.keys())},
            ).to_response()
        return t.invoke(arguments)
