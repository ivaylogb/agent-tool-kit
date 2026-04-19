"""Demo: knowledge-base search with semantic highlighting.

Compares the compressed search output with what a naive "return full document"
implementation would emit, showing the context savings end-to-end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_tool_kit import AuditLog  # noqa: E402
from examples.knowledge_base.corpus import all_documents  # noqa: E402
from examples.knowledge_base.tools import build_registry  # noqa: E402


def main() -> None:
    audit_log = AuditLog()
    registry = build_registry(audit_log=audit_log)

    print("===== Capability menu =====")
    for entry in registry.menu():
        print(f"- {entry['name']}: {entry['description']}")

    print("\n===== search_knowledge_base — compressed result =====")
    result = registry.execute(
        "search_knowledge_base",
        {
            "query": "How long do I have to return defective items?",
            "top_k": 3,
            "fragment_chars": 200,
        },
    )
    print(json.dumps(result, indent=2))

    print("\n===== Compression delta =====")
    naive_chars = sum(len(d.body) for d in all_documents())
    compressed_chars = len(json.dumps(result))
    print(f"Naive 'return all bodies' payload : {naive_chars:>5} chars")
    print(f"Compressed search response        : {compressed_chars:>5} chars")
    print(f"Reduction                         : {(1 - compressed_chars / naive_chars) * 100:.1f}%")

    print("\n===== fetch_document (when fragments aren't enough) =====")
    fetched = registry.execute("fetch_document", {"doc_id": "kb-006", "max_chars": 2000})
    print(json.dumps(fetched, indent=2))

    print("\n===== Schema-as-feedback: invalid doc_id pattern =====")
    bad = registry.execute("fetch_document", {"doc_id": "not-an-id", "max_chars": 1000})
    print(json.dumps(bad, indent=2))

    print("\n===== Audit metrics =====")
    print(json.dumps(audit_log.metrics(), indent=2))


if __name__ == "__main__":
    main()
