# E-commerce example — the fat tool pattern

This example demonstrates three things:

1. **Schema-as-feedback** — Pydantic input models with `Field(description=...)`
   and pattern constraints. Bad inputs become `invalid_input` errors that name
   the failing field, so the agent can self-correct without prompt iteration.
2. **Scope restrictions** — every tool carries an explicit `when_not_to_use`
   block. This is the single most effective lever against tool misrouting.
3. **Fat tools** — `process_return_request` wraps lookup → eligibility check →
   initiate-return into one call. The LLM can't reorder, skip, or misroute the
   chain, and the deterministic logic lives in code instead of in a prompt.

## Run the demo

```bash
pip install -e ../..
python run.py
```

The demo runs without an API key — it invokes tools directly via the registry,
showing the menu, a sample schema, success/error envelopes, the auto-escalation
branch of the fat tool, idempotency dedup, and an audit-log summary.

## Plug into an Anthropic agent

Each `Tool` in `tools.py` is a callable with a `tool_schema` attribute. To wire
into `agent_eval_loop.agent.runner.AgentRunner`:

```python
from agent_eval_loop.agent.runner import AgentRunner
from agent_eval_loop.models import AgentConfig
from examples.ecommerce.tools import build_registry, get_handlers

registry = build_registry()
config = AgentConfig(
    name="orders_v1",
    components={},
    model="claude-sonnet-4-6",
    tool_schemas=registry.all_schemas(),
)
runner = AgentRunner(config=config, tool_handlers=get_handlers())
runner.send_message("Where is order ORD-78234?")
```

For the full customer-support workflow (instructions, routines, scenarios),
swap `agent_eval_loop`'s `examples/customer_support/mocks.py` for
`get_handlers()` from this module — the tool names match.

**Error envelope key caveat**: customer_support's `components/tools/v1.yaml`
documents errors with a `code` field (e.g., `order_not_found`,
`outside_window`); toolkit envelopes use `category` (e.g., `not_found`,
`precondition_failed`). The agent generally infers the meaning, but for
exact prose alignment update the YAML's `errors:` block to reference
`category`/`message`/`suggested_action`, or wrap each handler with an
adapter that re-keys the envelope.
