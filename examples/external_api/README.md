# External API example — defensive wrapping

A typical anti-pattern: a tool calls an upstream API directly, lets exceptions
escape, and provides no retry or circuit protection. The agent crashes the
conversation or invents a fake recovery.

This example composes the toolkit's defensive primitives:

- **`RetryPolicy`** — jittered exponential backoff for transient failures.
- **`with_timeout`** — wall-clock cap on the upstream call.
- **`CircuitBreaker`** — fails fast after a threshold of consecutive errors,
  cools down, then half-opens to test recovery.
- **`defensive_call`** — composes the above and converts failure modes into
  structured `ToolError`s the LLM can reason about (`timeout`, `circuit_open`,
  `upstream_failure`).
- **`IdempotencyCache`** — same query within a session returns the cached
  reading; duplicate upstream calls don't happen.

## Run the demo

```bash
pip install -e ../..
python run.py
```

The demo drives the wrapped tool through four scenarios with a deterministic
fake API so every defensive behavior is visible in the output: retry
recovery, idempotency cache hit, timeout conversion, and circuit-breaker
tripping.
