"""Lightweight tool selection classifier.

Sits in front of the main agent: given a task description, asks a cheap model
(Haiku by default) to pick which tools from the registry's menu are likely
needed. The expensive executor model then runs with only the selected schemas
loaded.

This is the routing layer that makes progressive disclosure practical: the
classifier resolves the selection problem so the executor can focus on the
actual work.

The classifier is deliberately permissive on errors: if the model returns
malformed JSON or names tools that aren't registered, we fall back to the
full menu. A bad classifier should degrade gracefully into "load everything",
never into "block the request".
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from agent_tool_kit.registry import CapabilityRegistry

DEFAULT_CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"

CLASSIFIER_SYSTEM_PROMPT_TEMPLATE = (
    "You are a tool routing classifier. Given a task description and a menu of "
    "available tools, select the subset of tools that are likely to be needed "
    "to complete the task.\n\n"
    "Rules:\n"
    "- Be selective. Excluding irrelevant tools reduces context noise for the executor.\n"
    "- Include a tool if it might plausibly be called, even if you're not certain.\n"
    "- Output JSON only. No prose, no markdown.\n\n"
    "Output format: "
    "{{\"selected\": [\"tool_name\", ...], \"reasoning\": \"brief explanation\"}}\n\n"
    "Available tools:\n{menu}\n"
)


class ToolClassifier:
    """Selects relevant tools from a registry for a given task.

    Use it before constructing the agent's executor:

        classifier = ToolClassifier(registry)
        selected = classifier.select("The customer wants to track ORD-123.")
        runner = AgentRunner(
            config=cfg_with(tool_schemas=registry.schemas_for(selected)),
            tool_handlers=registry.handlers(selected),
        )
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        client: anthropic.Anthropic | None = None,
        model: str = DEFAULT_CLASSIFIER_MODEL,
        max_tokens: int = 400,
    ):
        self.registry = registry
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def select(self, task: str, *, max_tools: int | None = None) -> list[str]:
        """Return a list of tool names selected for the task.

        ``max_tools`` caps the result to the top-N selected names. ``None``
        (default) returns all selections.
        """
        menu = self.registry.menu()
        if not menu:
            return []
        menu_text = "\n".join(f"- {m['name']}: {m['summary']}" for m in menu)
        system = CLASSIFIER_SYSTEM_PROMPT_TEMPLATE.format(menu=menu_text)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": task}],
        )

        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        selected = self._parse_selection(text)
        if not selected:
            return self._fallback(max_tools)
        if max_tools is not None:
            selected = selected[:max_tools]
        return selected

    def _parse_selection(self, text: str) -> list[str]:
        """Best-effort extraction of the JSON selection from the model output."""
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data: Any = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict):
            return []
        raw = data.get("selected")
        if not isinstance(raw, list):
            return []
        return [n for n in raw if isinstance(n, str) and n in self.registry]

    def _fallback(self, max_tools: int | None) -> list[str]:
        """Conservative fallback: load everything (capped) when classification fails."""
        names = self.registry.names()
        if max_tools is not None:
            return names[:max_tools]
        return names
