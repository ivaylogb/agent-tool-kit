"""Demo: the e-commerce tools in action.

Runs without an Anthropic API key — exercises tools directly via the registry,
prints menu, schemas, results, and an audit-log summary. To plug into a live
agent, hand ``get_handlers()`` and ``build_registry().all_schemas()`` into
``agent_eval_loop.agent.runner.AgentRunner`` (see README for the snippet).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow the script to be run directly without installing the example package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_tool_kit import AuditLog, IdempotencyCache  # noqa: E402
from examples.ecommerce.tools import build_registry  # noqa: E402


def divider(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def main() -> None:
    audit_log = AuditLog()
    cache = IdempotencyCache()
    registry = build_registry(audit_log=audit_log, idempotency_cache=cache)

    divider("Capability menu (Level 1 — always loaded)")
    for entry in registry.menu():
        tags = f" [{', '.join(entry['tags'])}]" if entry["tags"] else ""
        print(f"- {entry['name']}: {entry['summary']}{tags}")

    divider("Schema for one tool (Level 2 — loaded on demand)")
    schemas = registry.schemas_for(["process_return_request"])
    print(json.dumps(schemas[0], indent=2)[:1200])
    print("...")

    divider("Direct invocation: lookup_order (success)")
    print(json.dumps(registry.execute("lookup_order", {"order_id": "ORD-78234"}), indent=2))

    divider("Schema-as-feedback: bad input → structured error")
    bad = registry.execute("lookup_order", {"order_id": "not-a-real-id"})
    print(json.dumps(bad, indent=2))

    divider("Fat tool: process_return_request — eligible order")
    eligible = registry.execute(
        "process_return_request",
        {"order_id": "ORD-78234", "reason": "wrong size"},
    )
    print(json.dumps(eligible, indent=2))

    # The fat tool is idempotent on order_id, so the next two scenarios use
    # fresh registries — otherwise the second call would return the first's
    # cached envelope. (That's the right behavior for production; for a demo
    # we want to show both code paths.)
    divider("Fat tool: process_return_request — outside window, defect → auto-escalate")
    fresh = build_registry()
    defect = fresh.execute(
        "process_return_request",
        {"order_id": "ORD-20100", "reason": "broken on arrival"},
    )
    print(json.dumps(defect, indent=2))

    divider("Fat tool: process_return_request — outside window, no defect → refused")
    fresh = build_registry()
    refused = fresh.execute(
        "process_return_request",
        {"order_id": "ORD-20100", "reason": "changed my mind"},
    )
    print(json.dumps(refused, indent=2))

    divider("Idempotency: duplicate cancel returns the same envelope")
    cancel_args = {"order_id": "ORD-91001", "reason": "ordered by mistake"}
    first = registry.execute("cancel_order", cancel_args)
    second = registry.execute("cancel_order", cancel_args)
    assert first == second
    print("first == second:", first == second)
    print("cancellation_id:", first["cancellation_id"])

    divider("Audit log summary")
    metrics = audit_log.metrics()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
