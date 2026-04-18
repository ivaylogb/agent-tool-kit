# agent-tool-kit

Production patterns and base classes for building reliable LLM tools.

> Most agent failures aren't prompt failures — they're tool failures. Bad schemas cause hallucinated parameters; thin wrappers create multi-step fragility; verbose outputs blow the context window; missing scope restrictions cause misrouting; unstructured errors leave the model unable to recover.

This package gives you the building blocks to design tools that absorb the variance of a stochastic caller (the LLM) and return structured feedback the model can use to self-correct.

It is the second repo in a portfolio of open-source patterns for production agent development. The first, `agent-eval-loop`, handles the **simulate → evaluate → improve** cycle. This one handles the **tools** the agent calls when it acts.

---

## Why tools first

The patterns in this repo come from production-tested work — see `tool-design-reference.md` and `docs/best-practices.md`. The headline ideas:

1. **The ACI paradigm** — a tool's contract is asymmetric. The caller is stochastic, the tool is deterministic. The tool must absorb the variance.
2. **Cognitive offloading** — every decision the agent makes costs tokens and is a failure point. Wrap multi-step workflows into one call.
3. **Schema as feedback** — Pydantic validation errors are teaching signals. A well-formed `invalid_input` envelope lets the model self-correct.
4. **Context compression** — tools should be good citizens of the context window. Return fragments, not documents. Project, don't dump.
5. **Capability registries** — at scale, two-level disclosure: cheap menu always loaded, full schemas on demand.
6. **Structured errors** — typed envelopes (`category`, `retryable`, `suggested_action`), never raw exceptions.
7. **Idempotency** — every action tool is safely re-callable.
8. **Observability** — every call logged. Every failure replayable.

---

## Install

```bash
pip install -e .
pip install -e ".[dev]"   # with pytest + ruff
```

Requires Python 3.11+, Pydantic v2, the Anthropic SDK.

---

## A 30-second tour

```python
from pydantic import BaseModel, Field
from agent_tool_kit import (
    CapabilityRegistry, ErrorCategory, ToolError, ToolException, tool,
)

class LookupOrderInput(BaseModel):
    order_id: str = Field(
        description="Order number formatted as 'ORD-XXXXX'.",
        pattern=r"^ORD-\d+$",
    )

@tool(
    name="lookup_order",
    description="Look up an order by order number.",
    when_not_to_use="Don't call this for tracking — use get_tracking_status.",
    input_model=LookupOrderInput,
)
def lookup_order(order_id: str) -> dict:
    order = ORDERS.get(order_id)
    if order is None:
        raise ToolException(ToolError(
            category=ErrorCategory.NOT_FOUND,
            message=f"Order {order_id!r} not found.",
            retryable=False,
            suggested_action="Ask the customer to check their confirmation email.",
        ))
    return order

# Two-level capability disclosure
registry = CapabilityRegistry([lookup_order])
print(registry.menu())                       # name + summary, ~50 tokens each
schemas = registry.schemas_for(["lookup_order"])  # full Anthropic schema

# Drop into agent_eval_loop's AgentRunner — handlers are callables with a tool_schema attr
from agent_eval_loop.agent.runner import AgentRunner
from agent_eval_loop.models import AgentConfig
runner = AgentRunner(
    config=AgentConfig(name="orders", components={}, tool_schemas=schemas),
    tool_handlers=registry.handlers(),
)
```

---

## Worked examples

Three self-contained demos in `examples/` that run without an API key:

| Example | Pattern |
|---|---|
| `examples/ecommerce/` | Fat tool — wraps lookup → eligibility check → initiate-return into one call. |
| `examples/knowledge_base/` | Context compression — semantic highlighting, field projection. ~80%+ token reduction. |
| `examples/external_api/` | Defensive composition — retries, timeouts, circuit breaker, structured errors. |

```bash
python examples/ecommerce/run.py
python examples/knowledge_base/run.py
python examples/external_api/run.py
```

Each prints menu/schema previews, exercises the success path, exercises the error paths, and ends with an audit-log summary so you can see the per-tool latency and error-rate breakdown.

---

## Compatibility with `agent-eval-loop`

Every `Tool` in this package is a callable with a `tool_schema` attribute. That's the exact contract `agent_eval_loop.agent.runner.AgentRunner.tool_handlers` expects, so you can mix and match without adapters.

The e-commerce example matches the tool **names** used by `../agent-eval-loop/examples/customer_support/`, so `examples.ecommerce.tools.get_handlers()` slots in mechanically as that example's `mocks.py`. **One caveat**: error envelopes use `category` (toolkit convention) where the customer_support YAML documents `code`. The agent typically copes, but if you want exact prose alignment, update either the YAML's `errors:` block or write a small re-keying adapter — see `CLAUDE.md` for details.

---

## Repo map

```
agent-tool-kit/
├── src/agent_tool_kit/        # Package source
│   ├── base.py                # Tool wrapper + @tool decorator
│   ├── errors.py              # ToolError, ToolException, ErrorCategory
│   ├── registry.py            # CapabilityRegistry (progressive disclosure)
│   ├── classifier.py          # ToolClassifier (Haiku-backed router)
│   ├── observability.py       # AuditLog, ToolCallRecord, replay, metrics
│   ├── idempotency.py         # IdempotencyCache
│   ├── defensive.py           # RetryPolicy, CircuitBreaker, defensive_call
│   └── compression.py         # project, highlight_fragments, truncate_text
├── examples/
│   ├── ecommerce/             # Fat tool pattern
│   ├── knowledge_base/        # Context compression
│   └── external_api/          # Defensive wrapping
├── tests/                     # 93 unit + integration tests
├── docs/best-practices.md     # Tool design best practices
├── tool-design-reference.md   # Original pattern reference
└── CLAUDE.md                  # For Claude Code iteration
```

---

## License

MIT.
