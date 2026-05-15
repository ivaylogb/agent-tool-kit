# Tool Design Reference — Key Patterns

This document covers tool-design patterns: the principles behind reliable agent tools — asymmetric tool contracts, cognitive offloading, schema-as-feedback, context compression, capability registries, and observability.

---

## 1. The ACI Paradigm (Agent-Computer Interaction)

The fundamental challenge: the caller (LLM) is stochastic, the executor (tool) is deterministic. The tool contract is asymmetric — the tool must absorb the variance of the caller.

In traditional software, a contract exists between two deterministic systems. If System A calls `getWeather("NYC")`, the output is consistent. But an agent might call `getWeather("NYC")`, `getWeather("New York")`, or hallucinate `getWeather("NYC", "Imperial")` even if the unit parameter doesn't exist.

Therefore: the tool must be hyper-robust. It cannot simply crash on bad input. It must provide feedback. If an agent calls a tool incorrectly, the tool's response is a teaching signal — a well-designed error message specifically crafted to guide the LLM back to the correct path.

Example: `"Error: 'Imperial' is not a valid unit. Please use 'C' or 'F'."` This turns the interaction into a constructive dialogue, allowing the agent to self-heal.

---

## 2. Cognitive Offloading — The Deterministic Chain Pattern

Every decision an agent makes requires token generation (cost + latency) and is a potential failure point. The principle: shift complexity from the reasoning engine to the tools.

### Fat Tool vs Thin Wrapper

**Thin Wrapper** — maps 1:1 to an API endpoint:
- `api.connect()`, `api.search(query)`, `api.disconnect()`
- Risk: agent might forget to connect, forget to disconnect, or fail to handle a timeout

**Fat Tool** — encapsulates a workflow:
- `perform_search_workflow(query)`
- Internally handles connection, retries, cleanup
- Agent calls one function, gets the result

### The Meta-Tool Pattern

A common application: RAG workflows. Instead of giving the agent a `vector_db_search` tool and a `summarize_text` tool (which requires the agent to orchestrate them correctly), build a single `get_knowledge(topic)` tool that internally:
1. Embeds the topic (deterministic)
2. Queries the vector DB (deterministic)
3. Retrieves top-k documents (deterministic)
4. Extracts key points via a cheaper LLM or heuristic (deterministic)
5. Returns a clean summary

The agent cannot forget to retrieve documents, nor accidentally retrieve from the wrong index. The workflow is fixed; only the parameters are dynamic.

---

## 3. Schema Validation as Cognitive Feedback

Pydantic is the de facto standard for tool input validation. Its value extends beyond type checking — it serves as a mechanism for cognitive feedback.

When an agent sends `{"age": "twenty"}` to an integer field, Pydantic raises: `Input should be a valid integer, unable to parse string as an integer.` In a well-designed framework, this error is caught and fed back to the agent as an observation. The agent reads the error, reasons "I made a mistake," and retries with `{"age": 20}`.

### Validation vs Sanitization (Defense in Depth)

- **Validation** asks: "Is this input correct?" (schema, types, ranges)
- **Sanitization** asks: "Is this input safe?" (SQL injection, malicious shell characters, path traversal)

Both are required. Validation happens first (Pydantic). Sanitization happens after validation but before execution (escaping, allow-listing).

### Structured Outputs and Explicit Refusal

Sometimes an agent should refuse to use a tool. The schema should include a refusal mechanism:
```json
{"refusal": "I am not authorized to prescribe medication."}
```
This lets the system programmatically detect the refusal instead of trying to parse it from unstructured text.

---

## 4. Context Compression and Semantic Highlighting

Tools must be "good citizens" of the context window. If a tool returns a 10MB JSON dump, it crashes the agent or truncates valuable history.

**Filtering:** A `search_users` tool should not return the entire user object (including hashed passwords, metadata, logs). Return a summarized view: name, id, email.

**Semantic Highlighting:** In retrieval scenarios, don't pass full documents. The tool performs the search, identifies relevant fragments, and constructs a synthetic document containing only those fragments. This "pre-digestion" offloads the burden of sifting through noise from the LLM to the retrieval algorithm.

Rule: tools return only the fields the LLM needs to reason about the next step. Everything else is noise.

---

## 5. The Capability Registry (Progressive Disclosure)

At scale, you can't load all tool descriptions into the context window. With 20+ tools, the agent might not reliably select the right one — the "10% hit rate problem."

The solution: a registry pattern with progressive disclosure.

### Two-level structure:
- **Level 1 (always loaded):** Tool name + one-line description. ~50-100 tokens per tool. The agent sees the menu.
- **Level 2 (loaded on demand):** Full schema, usage guidance, examples, scope restrictions. Only loaded when the agent selects a tool.

### How it works:
1. Agent receives a task
2. A lightweight classifier (cheap model like Haiku) determines which tools are relevant based on the task description
3. Full schemas for only those tools are loaded into context
4. Agent executes with the focused tool set

This is the same pattern as Claude's Skills architecture: store capabilities as files, load only the relevant one when needed. The agent starts lightweight and gets heavier only as needed.

### The routing layer:
Before the main agent processes each step, a lightweight classifier determines which tools are needed. The classifier uses the conversation context as its primary signal. This is division of labor between models: a cheap, fast model solves the selection problem so the expensive model can focus on the actual work.

---

## 6. Tool Observability

Tools are the agent's interface to the real world. When a tool is misused — called at the wrong time, with wrong parameters, or its output misinterpreted — the agent produces incorrect responses even if the prompt is perfect.

### What to track:
- **Audit trails:** Every tool call logged with inputs, outputs, latency, and errors. Replay capabilities allow engineers to reproduce failures without re-running the LLM.
- **Eval-driven detection:** Regression tests check tool selection (did the agent call the right tool?) and parameter extraction (did it extract correct parameters from the customer's message?).
- **Production monitoring:** Tool call patterns, error rates, and latency distributions tracked to detect systematic misuse patterns.

---

## 7. Five Tool Design Principles (Production-Tested)

1. **Compose deterministic flows in code, not in prompts.** When a task requires chaining multiple data sources, wrap the orchestration into a single composite tool.

2. **Tool descriptions are prompts.** Schema and docstring are part of the context. Include: natural-language description, typed input schema, expected output format, example invocations, and guidance on when NOT to call it.

3. **Minimize output surface area.** Return only the fields the LLM needs. A delivery-status tool returns carrier, status, and ETA — not the full shipment manifest.

4. **Design for idempotency.** Every action tool is safely re-callable. Enforce through idempotency keys derived from session + action parameters.

5. **Return structured errors, not exceptions.** Typed error objects with category, message, retryable flag. The LLM reasons about failures: retry on transient, request missing info, or escalate.
