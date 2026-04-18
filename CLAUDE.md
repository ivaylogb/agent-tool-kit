# CLAUDE.md — Agent Tool Kit

## What this is
Production patterns and base classes for building reliable LLM tools. This is the second repo in a portfolio that pairs with `../agent-eval-loop/`. The eval-loop repo handles **how the agent improves**; this repo handles **what the agent calls** when it acts.

Core thesis: most agent failures aren't prompt failures — they're tool failures. Bad schemas cause hallucinated parameters; thin wrappers create multi-step fragility; verbose outputs blow the context window; missing scope restrictions cause misrouting; unstructured errors leave the model unable to recover.

## Architecture
- `src/agent_tool_kit/base.py` — `Tool` wrapper + `@tool` decorator. Schema validation, idempotency, output filtering, audit hooks. Start here.
- `src/agent_tool_kit/errors.py` — `ToolError` envelope + `ErrorCategory`. Structured errors the LLM can reason about.
- `src/agent_tool_kit/registry.py` — `CapabilityRegistry`. Two-level disclosure: cheap menu + on-demand full schemas.
- `src/agent_tool_kit/classifier.py` — `ToolClassifier`. Haiku-backed router that selects relevant tools per task.
- `src/agent_tool_kit/observability.py` — `AuditLog` + `ToolCallRecord`. Append-only JSONL audit trail with replay and per-tool metrics.
- `src/agent_tool_kit/idempotency.py` — `IdempotencyCache`. Pluggable backend; in-memory default.
- `src/agent_tool_kit/defensive.py` — `RetryPolicy`, `CircuitBreaker`, `with_timeout`, `defensive_call`. Composable wrappers for unreliable upstreams.
- `src/agent_tool_kit/compression.py` — `project`, `highlight_fragments`, `truncate_text`. Context-compression helpers.
- `examples/ecommerce/` — Fat tool pattern. Schema-as-feedback, scope restrictions, deterministic chain composition.
- `examples/knowledge_base/` — Context compression. Field projection + semantic highlighting.
- `examples/external_api/` — Defensive composition. Retries, timeouts, circuit breaker, structured errors.
- `tests/` — 93 unit + integration tests. Smoke tests for every example.
- `docs/best-practices.md` — Tool-design best practices, distilled.
- `tool-design-reference.md` — The patterns this repo encodes (kept at root for quick reference during iteration).

## Key design decisions
- **Tools are callables with a `tool_schema` attribute.** This is the only contract — drops directly into `agent_eval_loop.agent.runner.AgentRunner.tool_handlers`. No registry required at the runner level.
- **Pydantic ValidationError → structured `invalid_input` envelope.** Validation isn't just type checking; the error message is a teaching signal. Each problem names the field, what's wrong, and what to do next.
- **Handlers raise `ToolException(ToolError(...))` for typed errors; the wrapper converts to envelopes.** Cleaner than returning dicts directly, and unified with the unexpected-exception path.
- **Idempotency keys are scoped per tool.** A 16-char SHA-256 over the validated input (or a subset of fields). Defaults to all fields; pass `idempotency_key_fields=[...]` to ignore cosmetic inputs (`reason`, free-form notes). For session scoping (which agent-eval-loop's docs phrase as "keys derived from session ID and action parameters"), construct a per-session `IdempotencyCache` instance — or, for distributed/shared caches, put `session_id` in the input model and include it in `idempotency_key_fields`. Both produce identical observed behavior; see `docs/best-practices.md` for which to use when.
- **`@tool` returns a `Tool` instance, not a function.** Trade: no `functools.wraps` magic, callers see a clear "this is a Tool" object. Pays off in introspection (the wrapper has properties, not just attributes hung on a function).
- **Defensive primitives compose; the toolkit doesn't pick a strategy.** `defensive_call(fn, retry_policy=..., timeout_seconds=..., circuit_breaker=...)` lets each tool author decide. Defaults are in `RetryPolicy()` and `CircuitBreaker()`, both opinionated but tunable.
- **Per-call latency timing lives in the wrapper, not the audit log.** `Tool.invoke` calls `audit_log.begin(call_id)` then `audit_log.record(call_id=...)`. Allows handlers without audit logs to skip timing entirely.
- **Examples build fresh state per `build_registry()` call.** No module-level caches that bleed across tests or sessions. Pass your own `AuditLog` / `IdempotencyCache` to share.

## Compatibility with agent-eval-loop
The `Tool` wrapper exposes the exact contract `AgentRunner` expects:

- Callable as `handler(**arguments)`. ✓
- Has `tool_schema` attribute (Anthropic dict format). ✓
- Returns dicts; never raises into the agent loop. ✓

To use a toolkit-built tool with `AgentRunner`:
```python
from agent_eval_loop.agent.runner import AgentRunner
from agent_eval_loop.models import AgentConfig
from examples.ecommerce.tools import build_registry, get_handlers

reg = build_registry()
config = AgentConfig(name="orders", components={}, tool_schemas=reg.all_schemas())
runner = AgentRunner(config=config, tool_handlers=get_handlers())
```

The e-commerce example matches the tool **names** used by `../agent-eval-loop/examples/customer_support/` so `get_handlers()` slots in mechanically as that example's `mocks.py`. **Caveat on error envelope keys**: agent-eval-loop's `customer_support/components/tools/v1.yaml` documents errors with a `code` field (`order_not_found`, `outside_window`, etc.). Toolkit envelopes use `category` instead, with toolkit-standard category strings (`not_found`, `precondition_failed`, etc.). The agent will usually cope — both names convey the same meaning — but if the YAML's "Errors:" prose is load-bearing for the agent, update either the YAML to describe `category`/`message`/`suggested_action`, or write a small adapter that re-keys envelopes back to `code`. The toolkit's keys are the better surface; we recommend updating the YAML.

## Best practices encoded
- **Schema-as-feedback** — Pydantic `Field(description=...)` and constraints become the agent's self-correction signal. Bad inputs return `invalid_input` with the failing field named.
- **Scope restrictions** — every tool description carries a `when_not_to_use` block. Without it, models call the most semantically similar tool regardless of correctness.
- **Fat tools** — `process_return_request` wraps lookup → eligibility check → initiate-return into one call. The LLM can't reorder, skip, or misroute.
- **Context compression** — `search_knowledge_base` returns `fragments`, never bodies. ~80%+ context reduction vs naive "return everything".
- **Defensive composition** — every external-API tool wraps the upstream in `defensive_call`. Failures become typed envelopes the LLM can reason about (`timeout`, `circuit_open`, `upstream_failure`).
- **Idempotency** — every action tool that has side effects is idempotent on the action's identifying fields. Duplicate calls return the cached envelope, not a duplicate side effect.
- **Progressive disclosure** — `CapabilityRegistry.menu()` returns one-line summaries; `schemas_for(names)` loads the full schemas only for the selected subset.

## Commands
```bash
pip install -e .                     # Install
pip install -e ".[dev]"              # With pytest + ruff
python examples/ecommerce/run.py     # Demo: fat tool pattern
python examples/knowledge_base/run.py  # Demo: context compression
python examples/external_api/run.py    # Demo: defensive wrapping
ruff check src/ examples/ tests/     # Lint
pytest                               # 93 unit + integration tests
```

## Conventions
- Python 3.11+. `from __future__ import annotations` everywhere — annotations stay as strings unless the wrapper resolves them.
- Pydantic v2. `model_json_schema()` strips `title` keys before sending to Anthropic.
- Anthropic SDK only used by `ToolClassifier`. Tests stub the client; no live API calls in CI.
- UTC timestamps via `datetime.now(timezone.utc)`.
- Ruff with `E,F,I,N,W` selected, line length 100. `N818` ignored — `ToolException` is a deliberate name.
- No emojis in code or docs.

## What to build next
- A Redis-backed `IdempotencyCache` implementation for distributed deployments.
- A Streamlit/Rich-based audit-log inspector that loads `audit.jsonl` and shows per-tool failure heatmaps.
- A `ToolContext` object the wrapper threads into handlers (session_id, deadline, audit hook) so handlers can use it without closure tricks.
- A composability example: KB search tool used inside an e-commerce return workflow ("look up the policy for this defect class first").
- Integration tests that drive a full `AgentRunner` conversation against real Anthropic — gated on `ANTHROPIC_API_KEY`.
- A capability-registry recipe doc showing classifier + executor split with concrete numbers.
