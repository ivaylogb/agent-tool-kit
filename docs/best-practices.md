# Best Practices for LLM Tools

Lessons from building tools that production agents call millions of times. These principles are domain-agnostic — they apply whether the agent is doing customer support, code review, or anything else.

---

## The contract is asymmetric

The caller is stochastic. The tool is deterministic. The contract is not symmetric.

In traditional software two deterministic systems agree on a contract. In agent systems, the LLM might call your tool with `getWeather("NYC")`, `getWeather("New York")`, or `getWeather("NYC", unit="Imperial")` even when the `unit` parameter doesn't exist. The tool absorbs the variance, or the agent fails.

Practical implications:

- **Validate inputs strictly, but turn validation failures into teaching signals.** A `ValidationError` becomes `{"error": {"category": "invalid_input", "message": "order_id: String should match pattern '^ORD-\\d+$'", ...}}` — the agent reads it, sees what's wrong, and self-corrects.
- **Never let exceptions escape into the agent loop.** A Python traceback in the context window is worse than useless: it wastes tokens and the model can't act on it. Wrap everything; convert to typed envelopes.
- **Idempotent action tools.** The agent can and will retry. If `create_return` runs twice, the second call must return the first's result, not a duplicate side effect.

---

## Schema design is prompt design

The schema and the docstring are part of the agent's context window. Treat them with the same rigor as any other prompt component.

**Required ingredients in every tool:**

1. Natural-language description of what the tool does.
2. Typed input schema — every field has a `description=`, type, and (where useful) constraints (`pattern=`, `min_length=`, `Literal[...]`).
3. Expected output shape — what the agent will see on success.
4. **When NOT to use it.** This is the single most-missed piece. Without it, the model picks the most semantically similar tool regardless of correctness. With it, misrouting drops sharply.

**Example anti-pattern (no scope restriction):**
```yaml
- name: search_knowledge_base
  description: "Search the knowledge base."
```
Result: this gets called for "where is order ORD-123" because "search" is semantically close to "look up".

**With scope restriction:**
```yaml
- name: search_knowledge_base
  description: "Search the knowledge base for general policy questions."
  when_not_to_use: |
    Do not use for order-specific questions (status, tracking, individual items)
    — those need the order tools, not the policy KB.
```

In this toolkit, `when_not_to_use` is a first-class parameter on `@tool` and is appended to the description sent to Anthropic.

---

## Fat tools beat prompt orchestration

When a task requires chaining (check status → verify eligibility → perform action → confirm), wrap the entire chain into one composite tool. The LLM makes one call; the deterministic logic lives in code.

Why:
- **Lower latency.** One round-trip instead of three.
- **No sequential-dependency bugs.** The model can't forget step 2 or call them out of order.
- **Easier to test.** Workflows are pure functions of their inputs.
- **Easier to reason about.** Failures are in one place.

The e-commerce example's `process_return_request` demonstrates this. The naive design — `lookup_order` then `initiate_return` — leaves the agent to remember the eligibility check and to know which path to take when the order's outside the window. The fat tool encodes the routing in code, including auto-escalation for out-of-window defects.

Reserve thin tools for cases where the agent legitimately needs the intermediate result to make a decision the workflow can't predict.

---

## Minimize output surface area

A tool that returns a 10MB JSON blob crashes the conversation or truncates valuable history. Return only the fields the LLM needs to reason about the next step.

Two techniques in this toolkit:

1. **Field projection** — `project(record, ["id", "title", "score"])` keeps a known subset, drops everything else. Use as `output_filter` on the `Tool`, or inline in the handler.
2. **Semantic highlighting** — for retrieval, run the search, identify relevant fragments, return a synthetic document containing only the matching spans. `highlight_fragments(text, query_terms, fragment_chars=240, max_fragments=3)`.

The knowledge-base example demonstrates the second technique end-to-end. Empirical reduction for a small corpus answering one query: 80%+ versus a naive "return every body" implementation. The savings grow with corpus size.

A useful test: if your tool returns a field the agent never uses, delete it. If it never gets deleted, the field is overhead.

---

## Structured errors, not exceptions

Every error is a typed envelope:

```json
{
  "error": {
    "category": "precondition_failed",
    "message": "Order is 45 days old; the return window is 30 days.",
    "retryable": false,
    "suggested_action": "If the reason indicates a defect, escalate to a human agent. Otherwise, explain the return-window policy.",
    "details": {"age_days": 45}
  }
}
```

The categories the LLM most often needs to distinguish: `invalid_input`, `not_found`, `unauthorized`, `precondition_failed`, `timeout`, `upstream_failure`, `rate_limited`, `circuit_open`, `conflict`, `internal`, `refusal`. The toolkit provides them as `ErrorCategory`; extend by passing a string-coercible value.

**Why `suggested_action` matters:** the LLM doesn't always know what to do with an error. `"Order outside return window"` is informational. `"If the reason indicates a defect, escalate to a human agent. Otherwise, explain the return-window policy."` is actionable. The cost is one short string per error path; the benefit is the agent's recovery rate.

---

## Idempotency is non-negotiable for action tools

Agents retry. Context windows truncate. Models occasionally emit a duplicate `tool_use` block. If `create_return` runs twice, you don't want two return labels.

The pattern:

```python
@tool(
    ...,
    idempotent=True,
    idempotency_cache=cache,
    idempotency_key_fields=["order_id"],   # ignore cosmetic fields like `reason`
)
def initiate_return(order_id: str, reason: str) -> dict:
    ...
```

The cache key is a hash over the named fields. On a duplicate call, the first result is returned without re-executing. Errors are not cached — only successful side effects.

`idempotency_key_fields` matters: `reason` is free-form text the agent might phrase slightly differently between calls. Excluding it from the key means the dedup actually works.

### Session scoping — two equivalent patterns

The `agent-eval-loop` best-practices document phrases this as "keys derived from the session ID and action parameters." The toolkit achieves the same outcome by a different mechanism — pick whichever fits your deployment:

1. **Per-session cache instance (recommended for in-memory).** Construct a fresh `IdempotencyCache` per session and pass it into your tools. Two sessions never see each other's cache because they hold independent instances. This is what `examples/ecommerce/build_registry()` does. Simple, no key manipulation, no cross-session leakage.

2. **Session ID in the key (recommended for distributed caches).** When the cache backend is shared across processes (Redis, Memcached), put `session_id` in the input model and include it in `idempotency_key_fields`. Then a single shared cache stores per-(session, action) entries. Use this when you can't afford one cache instance per session.

Both produce identical observed behavior; the choice is operational.

---

## Defensive composition for unreliable upstreams

Tools that call external APIs need three things, composable:

- **Retries** — `RetryPolicy(max_attempts=3, base_delay=0.5, multiplier=2.0, jitter=0.25)`. Jitter avoids thundering herds when many agents fail at once.
- **Timeouts** — wall-clock cap. The toolkit uses a worker thread (the upstream call continues running on timeout, since there's no safe way to interrupt arbitrary sync code in Python).
- **Circuit breaker** — `CircuitBreaker(failure_threshold=5, cooldown_seconds=30)`. After N consecutive failures, the breaker opens and calls fail fast with `circuit_open`. After the cooldown it half-opens for a trial; success closes it, failure re-opens.

`defensive_call(fn, retry_policy=..., timeout_seconds=..., circuit_breaker=...)` composes the three and converts the failure modes the agent should reason about (timeout, circuit open, exhausted retries) into typed `ToolException(ToolError(...))`. The `Tool` wrapper catches and converts to envelopes.

---

## Capability registries scale tool catalogues

At scale you can't load every tool's full schema into context. With 20+ tools selection accuracy degrades — the "10% hit rate problem".

Two-level structure:

- **Menu (always loaded)** — `name + summary + tags`, ~50–100 tokens per tool. The agent sees the catalogue.
- **Schemas (on demand)** — full Anthropic-format definitions, loaded only for the subset relevant to the current task.

The selection step is a separate, cheap classifier (`ToolClassifier` uses Haiku by default). It reads the menu and the task, returns a list of selected tool names. The expensive executor model then runs with only those schemas loaded.

```python
classifier = ToolClassifier(registry)
selected = classifier.select("Customer wants to track ORD-123")
runner = AgentRunner(
    config=AgentConfig(name="x", components={}, tool_schemas=registry.schemas_for(selected)),
    tool_handlers=registry.handlers(selected),
)
```

A bad classifier should degrade gracefully into "load everything", never into "block the request". The toolkit's classifier returns the full menu when JSON parsing fails or selection comes back empty.

---

## Observability or you're flying blind

Tools are the agent's interface to the real world. When a tool is misused — wrong time, wrong parameters, wrong interpretation of output — the agent produces incorrect responses even with a perfect prompt.

Track:

- **Audit trails** — every call, with inputs, outputs, latency, error envelope. Append-only JSONL. The toolkit's `AuditLog` flushes per record so a crash mid-conversation still leaves a complete trail.
- **Replay** — `AuditLog.replay(path)` iterates historical entries. Reproduce failures without re-running the LLM.
- **Per-tool metrics** — `audit_log.metrics()` returns per-tool call counts, error rates, average latency. Surface these in your monitoring; alert on systematic increases.
- **Eval-driven detection** — tool selection and parameter extraction are first-class eval categories in `agent-eval-loop`. Did the agent call the right tool? Did it extract the right parameters?

---

## Version your tools like prompts

Treat the schema and docstring of each tool as a versioned artifact. When a description changes, that's a new version that needs the same eval gate as a new system-prompt revision.

In the toolkit, this is implicit — each tool is a Python object with a frozen description. In the `agent-eval-loop` framework's terms, your tool descriptions are a `ComponentType.TOOLS` component; bumping a version is bumping the file. The eval loop's regression suite catches descriptions that broke previously passing cases.

---

## Common anti-patterns to delete on sight

- **Returning `{"status": "ok"}` with no other detail on success.** The agent has nothing to confirm to the user. Return the receipt: IDs, timelines, what to expect.
- **Returning a stack trace as the error message.** The model can't act on it. Return a category and a suggested action.
- **Tools that branch on caller-supplied "mode" strings.** Splits the contract; doubles the schemas to test. Two tools beat one tool with a `mode` parameter.
- **Tools that require the agent to first call a "session start" tool.** Hide it inside the tool's first call; emit a structured `precondition_failed` if the session is gone.
- **Tools that return the same shape for success and failure.** `{"success": True, "data": ...}` vs `{"success": False, "error": ...}` is fine, but inconsistent shapes — `data` sometimes a dict, sometimes a string — are a parsing nightmare.
- **`when_not_to_use` blocks that just say "use carefully".** Be specific: name the other tool that's the right call, or name the precondition that must hold.

---

## When to break these rules

These are defaults, not commandments. A few legitimate exceptions:

- **Returning more fields than strictly needed when the schema documents them.** If the agent's downstream reasoning is sensitive to a field the LLM doesn't always cite, it's still load-bearing — keep it.
- **Skipping the circuit breaker for read-only fast endpoints.** A breaker on a 1ms in-memory cache is overhead, not safety.
- **Letting a tool be slightly less idempotent if the cost of duplicate side effects is genuinely zero.** Sending an idempotent log line, for example.

The rule: every exception should be a deliberate, documented choice. If you can't articulate why you broke the pattern, you're cutting a corner that will cost you later.
