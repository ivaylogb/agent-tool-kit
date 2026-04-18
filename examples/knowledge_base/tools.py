"""Knowledge base tools — context compression in practice.

The naive design returns full documents on every search hit. That blows the
context window and forces the LLM to sift through noise. This implementation
demonstrates two compression techniques:

1. **Field projection** — the search tool returns ``doc_id``, ``title``,
   ``score``, ``tags``, and ``fragments``. Not the body. Not internal IDs.
   Just the minimum the LLM needs to cite a source or decide on a follow-up.
2. **Semantic highlighting** — instead of returning the matching document
   body, the tool extracts ~240-character fragments around term matches.
   The LLM sees the spans that justify the match, in document order.

The LLM-facing schema documents both behaviors, so the agent knows it should
call ``fetch_document`` if it genuinely needs the full body of a hit.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from agent_tool_kit import (
    AuditLog,
    CapabilityRegistry,
    ErrorCategory,
    Tool,
    ToolError,
    ToolException,
    highlight_fragments,
    project,
    tool,
    truncate_text,
)

from .corpus import all_documents, by_id

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "how", "i", "if", "in", "is", "it", "of", "on", "or", "the", "to", "what",
    "when", "where", "who", "why", "you", "your", "with", "about",
})


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _meaningful_terms(query: str) -> list[str]:
    return [t for t in _tokenize(query) if t not in _STOPWORDS]


def _score(doc_text: str, terms: list[str]) -> float:
    """Simple normalized term-overlap score."""
    if not terms:
        return 0.0
    tokens = Counter(_tokenize(doc_text))
    if not tokens:
        return 0.0
    overlap = sum(tokens.get(t, 0) for t in terms)
    return round(overlap / sum(tokens.values()) * 100.0, 4)


# -- Pydantic input models ----------------------------------------------------

class SearchInput(BaseModel):
    query: str = Field(
        description="The natural-language query. Stopwords are dropped automatically.",
        min_length=2,
    )
    top_k: int = Field(
        3,
        description="Maximum number of results to return. The tool caps at 5.",
        ge=1,
        le=5,
    )
    fragment_chars: int = Field(
        240,
        description="Characters per highlighted fragment around each match.",
        ge=80,
        le=600,
    )


class FetchInput(BaseModel):
    doc_id: str = Field(
        description="The document ID, e.g., 'kb-003'. Get this from search_knowledge_base results.",
        pattern=r"^kb-\d+$",
    )
    max_chars: int = Field(
        2000,
        description="Hard cap on returned body length; longer bodies are truncated.",
        ge=200,
        le=8000,
    )


# -- Tools --------------------------------------------------------------------

def _make_search(audit: AuditLog) -> Tool:
    @tool(
        name="search_knowledge_base",
        description=(
            "Search ShopFast's knowledge base. Returns the top-k matching documents "
            "as compressed result cards: doc_id, title, score, tags, and short text "
            "fragments highlighting the parts that matched the query. Full document "
            "bodies are NOT returned — call fetch_document(doc_id) if you need one.\n\n"
            "WHEN TO USE: Customer asks a policy question (returns, shipping, "
            "warranty, account) where the answer is the same for every customer."
        ),
        when_not_to_use=(
            "Do not use this tool for order-specific questions (status, tracking, "
            "individual items) — those need order tools, not the policy KB. Do not "
            "use to answer questions about pricing, promotions, or product "
            "availability — those live in different systems. Do not chain "
            "fetch_document afterward unless a fragment is genuinely insufficient."
        ),
        input_model=SearchInput,
        tags=["knowledge", "search"],
        audit_log=audit,
    )
    def search_knowledge_base(query: str, top_k: int, fragment_chars: int) -> dict[str, Any]:
        terms = _meaningful_terms(query)
        if not terms:
            raise ToolException(ToolError(
                category=ErrorCategory.INVALID_INPUT,
                message="Query contained only stopwords; nothing to search for.",
                retryable=True,
                suggested_action="Re-issue the call with at least one content word.",
            ))

        scored: list[tuple[float, Any]] = []
        for doc in all_documents():
            s = _score(doc.title + "\n" + doc.body, terms)
            if s > 0:
                scored.append((s, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        results: list[dict[str, Any]] = []
        for score, doc in top:
            fragments = highlight_fragments(
                doc.body,
                terms,
                fragment_chars=fragment_chars,
                max_fragments=3,
            )
            if not fragments:
                # Title-only match: synthesize a single fragment from the title.
                fragments = [doc.title]
            results.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "tags": list(doc.tags),
                "score": score,
                "fragments": fragments,
            })

        return {
            "query": query,
            "match_count": len(scored),
            "returned": len(results),
            "results": results,
        }

    return search_knowledge_base


def _make_fetch(audit: AuditLog) -> Tool:
    @tool(
        name="fetch_document",
        description=(
            "Fetch the full body of one knowledge-base document. Returned body is "
            "capped at max_chars characters with an elision marker if truncated."
        ),
        when_not_to_use=(
            "Do not call without first running search_knowledge_base — guessing "
            "doc_ids wastes turns. Don't call to answer a question that the "
            "search-result fragments already answered. Don't fetch multiple docs "
            "speculatively — one fragment-driven follow-up is usually enough."
        ),
        input_model=FetchInput,
        tags=["knowledge"],
        audit_log=audit,
    )
    def fetch_document(doc_id: str, max_chars: int) -> dict[str, Any]:
        doc = by_id(doc_id)
        if doc is None:
            raise ToolException(ToolError(
                category=ErrorCategory.NOT_FOUND,
                message=f"No document with id {doc_id!r} exists.",
                retryable=False,
                suggested_action="Run search_knowledge_base first to discover valid doc_ids.",
            ))
        body = truncate_text(doc.body, max_chars)
        record = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "tags": list(doc.tags),
            "body": body,
            "truncated": len(body) < len(doc.body),
            "full_length": len(doc.body),
        }
        # The handler builds the projection explicitly so schema and response
        # stay in lockstep — adding a field here doesn't accidentally leak it
        # to the LLM unless ``project`` is updated too.
        return project(record, ["doc_id", "title", "tags", "body", "truncated", "full_length"])

    return fetch_document


def build_tools(
    audit_log: AuditLog | None = None,
) -> tuple[list[Tool], AuditLog]:
    """Construct fresh tools wired to the given audit log."""
    audit = audit_log if audit_log is not None else AuditLog()
    tools = [_make_search(audit), _make_fetch(audit)]
    return tools, audit


def build_registry(audit_log: AuditLog | None = None) -> CapabilityRegistry:
    tools, _ = build_tools(audit_log)
    return CapabilityRegistry(tools)


def get_handlers(audit_log: AuditLog | None = None) -> dict[str, Tool]:
    tools, _ = build_tools(audit_log)
    return {t.name: t for t in tools}
