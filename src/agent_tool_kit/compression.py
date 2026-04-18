"""Context-compression helpers for tool outputs.

Tools are good citizens of the context window. A tool that returns a 10MB JSON
dump crashes the agent or truncates valuable history. Two patterns this module
supports:

1. **Filtering** — return only the fields the LLM needs to reason about.
   Use ``project`` / ``project_many`` to drop everything else (internal IDs,
   metadata, audit fields).

2. **Semantic highlighting** — for retrieval scenarios, don't pass full
   documents. Run the search, identify relevant fragments, and return a
   pre-digested view. ``highlight_fragments`` does this for keyword queries.

Both pattern implementations are deliberately minimal — they're meant to be
slotted into a tool's ``output_filter`` or used directly inside the handler.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence


def project(record: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    """Return only the named fields from a record, in the requested order.

    Missing fields are silently dropped — the projection is best-effort, since
    the LLM-facing schema should already document which fields are returned.
    """
    field_list = list(fields)
    return {k: record[k] for k in field_list if k in record}


def project_many(
    records: Iterable[dict[str, Any]],
    fields: Iterable[str],
) -> list[dict[str, Any]]:
    """Apply ``project`` to each record."""
    fl = list(fields)
    return [project(r, fl) for r in records]


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """Truncate ``text`` to at most ``max_chars`` characters, including suffix."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(suffix):
        return suffix[:max_chars]
    return text[: max_chars - len(suffix)] + suffix


def highlight_fragments(
    text: str,
    query_terms: Sequence[str],
    *,
    fragment_chars: int = 240,
    max_fragments: int = 3,
) -> list[str]:
    """Extract short fragments of ``text`` around query-term matches.

    Returns a list of short strings (each up to ``fragment_chars`` characters,
    plus elision markers) in document order. Overlapping fragments are merged
    by skipping later matches that already fall inside an emitted span.

    Use this in retrieval tools to return only the spans that justify the
    match — pre-digesting noise out of the LLM's context.
    """
    if not text or not query_terms or fragment_chars <= 0 or max_fragments <= 0:
        return []
    pattern = "|".join(re.escape(t) for t in query_terms if t)
    if not pattern:
        return []
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not matches:
        return []
    fragments: list[str] = []
    used_spans: list[tuple[int, int]] = []
    half = fragment_chars // 2
    for m in matches:
        if len(fragments) >= max_fragments:
            break
        center = (m.start() + m.end()) // 2
        start = max(0, center - half)
        end = min(len(text), center + half)
        if any(s < end and start < e for s, e in used_spans):
            continue
        used_spans.append((start, end))
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        fragment = text[start:end].strip()
        fragments.append(f"{prefix}{fragment}{suffix}")
    return fragments
