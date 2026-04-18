"""The Tool wrapper and ``@tool`` decorator.

A ``Tool`` is a callable that:

1. Validates input via Pydantic. Validation errors become structured
   ``ToolError`` envelopes the LLM can read and self-correct from.
2. Optionally consults an idempotency cache so duplicate calls return the
   first result instead of re-executing the side effect.
3. Executes the underlying handler. ``ToolException`` shortcuts to a typed
   error envelope; unexpected exceptions are caught and reported as
   ``ErrorCategory.INTERNAL`` rather than escaping into the agent loop.
4. Optionally projects the result through an ``output_filter`` (compression).
5. Records the call to an audit log if one is attached.

The wrapper exposes a ``tool_schema`` attribute in the Anthropic API format,
so it slots directly into anything that expects ``tool_handlers[name](...)``
plus a schema sidecar — including ``agent_eval_loop.agent.runner.AgentRunner``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from agent_tool_kit.errors import ErrorCategory, ToolError, ToolException
from agent_tool_kit.idempotency import IdempotencyCache
from agent_tool_kit.observability import AuditLog


def _strip_titles(obj: Any) -> Any:
    """Recursively drop ``title`` keys from a Pydantic-generated JSON Schema.

    Pydantic emits ``title`` for every field; Anthropic accepts but doesn't
    need them, and they're noise the LLM has to skim past. Stripping them
    shaves tokens and reduces visual clutter in the schema preview.
    """
    if isinstance(obj, dict):
        return {k: _strip_titles(v) for k, v in obj.items() if k != "title"}
    if isinstance(obj, list):
        return [_strip_titles(x) for x in obj]
    return obj


def _is_error_envelope(result: Any) -> bool:
    return (
        isinstance(result, dict)
        and "error" in result
        and isinstance(result["error"], dict)
    )


class Tool:
    """Schema-validated, observable, idempotency-aware tool wrapper.

    Construct via the ``@tool`` decorator in the common case. Direct
    construction is supported for tools you want to wire programmatically
    (e.g., factory functions that bind a tool to a specific session).
    """

    def __init__(
        self,
        handler: Callable[..., Any],
        *,
        name: str,
        description: str,
        input_model: type[BaseModel],
        when_not_to_use: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        idempotent: bool = False,
        idempotency_key_fields: list[str] | tuple[str, ...] | None = None,
        idempotency_cache: IdempotencyCache | None = None,
        audit_log: AuditLog | None = None,
        output_filter: Callable[[Any], Any] | None = None,
    ):
        if not name:
            raise ValueError("Tool name is required.")
        if not isinstance(input_model, type) or not issubclass(input_model, BaseModel):
            raise TypeError("input_model must be a Pydantic BaseModel subclass.")
        if idempotent and idempotency_cache is None:
            raise ValueError(
                "idempotent=True requires an idempotency_cache. "
                "Pass IdempotencyCache() or a custom backend."
            )
        self.handler = handler
        self.name = name
        self.description = description.strip()
        self.input_model = input_model
        self.when_not_to_use = when_not_to_use.strip() if when_not_to_use else None
        self.tags = tuple(tags) if tags else ()
        self.idempotent = idempotent
        self.idempotency_key_fields = tuple(idempotency_key_fields or ())
        self.idempotency_cache = idempotency_cache
        self.audit_log = audit_log
        self.output_filter = output_filter
        self._tool_schema = self._build_tool_schema()

    # ------------------------------------------------------------------ schema

    @property
    def tool_schema(self) -> dict[str, Any]:
        """Anthropic-API-compatible tool schema dict.

        AgentRunner picks this up via ``getattr(handler, 'tool_schema', None)``
        and forwards it to the Messages API as the tool definition.
        """
        return self._tool_schema

    @property
    def menu_entry(self) -> dict[str, Any]:
        """Lightweight tool metadata for progressive disclosure (always loaded).

        Returns name + one-line summary + tags. ~50-100 tokens per tool —
        cheap enough to keep the entire registry's menu in the system prompt
        even for large tool catalogues.
        """
        return {
            "name": self.name,
            "summary": self._summary(),
            "tags": list(self.tags),
        }

    def _summary(self) -> str:
        first_line = self.description.split("\n", 1)[0].strip()
        if len(first_line) > 160:
            first_line = first_line[:157] + "..."
        return first_line

    def _build_tool_schema(self) -> dict[str, Any]:
        raw_schema = self.input_model.model_json_schema()
        cleaned = _strip_titles(raw_schema)
        # Anthropic requires "type": "object" at the top level.
        cleaned.setdefault("type", "object")
        full_description = self.description
        if self.when_not_to_use:
            full_description = (
                f"{full_description}\n\nWHEN NOT TO USE:\n{self.when_not_to_use}"
            )
        return {
            "name": self.name,
            "description": full_description,
            "input_schema": cleaned,
        }

    # ----------------------------------------------------------------- invoke

    def __call__(self, **kwargs: Any) -> Any:
        """AgentRunner-compatible entry point: handler(**arguments)."""
        return self.invoke(kwargs)

    def invoke(self, raw_input: dict[str, Any]) -> Any:
        """Run the tool against a raw input dict; return result or error envelope.

        Wrapped in a try/finally so that a leak from any post-handler step
        (output filter, idempotency cache write) cannot leave a stale entry
        in ``audit_log._start_times`` and cannot let an exception escape into
        the agent loop. Every exit path produces a dict.
        """
        call_id = uuid.uuid4().hex
        if self.audit_log is not None:
            self.audit_log.begin(call_id)

        recorded = False
        try:
            # 1. Validate input — schema errors become teaching signals.
            try:
                parsed = self.input_model.model_validate(raw_input)
            except ValidationError as e:
                err = self._validation_error_to_tool_error(e)
                envelope = err.to_response()
                self._record_audit(call_id, raw_input, envelope, error=err)
                recorded = True
                return envelope

            # 2. Idempotency: short-circuit on cache hit.
            idem_key: str | None = None
            if self.idempotent and self.idempotency_cache is not None:
                idem_key = self._compute_idempotency_key(parsed)
                cached = self.idempotency_cache.get(self.name, idem_key)
                if cached is not None:
                    self._record_audit(call_id, raw_input, cached, idempotent_hit=True)
                    recorded = True
                    return cached

            # 3. Execute handler.
            try:
                result = self._call_handler(parsed)
            except ToolException as te:
                envelope = te.error.to_response()
                self._record_audit(call_id, raw_input, envelope, error=te.error)
                recorded = True
                return envelope
            except Exception as e:  # noqa: BLE001 — last-resort guardrail
                err = ToolError(
                    category=ErrorCategory.INTERNAL,
                    message=f"Unhandled tool error: {type(e).__name__}: {e}",
                    retryable=False,
                    suggested_action=(
                        "This is a bug in the tool implementation. Escalate to a human."
                    ),
                    details={"exception_type": type(e).__name__},
                )
                envelope = err.to_response()
                self._record_audit(call_id, raw_input, envelope, error=err)
                recorded = True
                return envelope

            # 4. Optional output filter (post-execution compression).
            #    Skip filtering for error envelopes the handler returned directly —
            #    filters are written for the success shape and shouldn't see errors.
            if self.output_filter is not None and not _is_error_envelope(result):
                try:
                    result = self.output_filter(result)
                except Exception as e:  # noqa: BLE001 — filter is third-party-ish
                    err = ToolError(
                        category=ErrorCategory.INTERNAL,
                        message=(
                            f"output_filter raised: {type(e).__name__}: {e}"
                        ),
                        retryable=False,
                        suggested_action=(
                            "The tool's output filter has a bug. The raw handler "
                            "result is unavailable; escalate to a human."
                        ),
                        details={"exception_type": type(e).__name__},
                    )
                    envelope = err.to_response()
                    self._record_audit(call_id, raw_input, envelope, error=err)
                    recorded = True
                    return envelope

            # 5. Cache successful results for idempotent tools.
            if (
                self.idempotent
                and self.idempotency_cache is not None
                and idem_key is not None
                and not _is_error_envelope(result)
            ):
                self.idempotency_cache.put(self.name, idem_key, result)

            self._record_audit(call_id, raw_input, result)
            recorded = True
            return result
        finally:
            # Belt-and-braces: if some unforeseen path skips _record_audit (which
            # is what pops ``_start_times``), drop the stale entry here so the
            # log doesn't accumulate orphaned timers under repeated misuse.
            if (
                not recorded
                and self.audit_log is not None
                and call_id in self.audit_log._start_times
            ):
                self.audit_log._start_times.pop(call_id, None)

    # ------------------------------------------------------------ internals

    def _call_handler(self, parsed: BaseModel) -> Any:
        """Dispatch to the handler with the right calling convention.

        If the handler takes a single positional/keyword param annotated as
        a Pydantic model, pass the parsed model in directly. Otherwise, spread
        the validated fields as keyword arguments. This lets handler authors
        choose either style without having to declare it explicitly.

        Annotations may be strings (PEP 563 / ``from __future__ import
        annotations``); we resolve via ``eval_str=True`` and fall back to
        a string-name comparison if evaluation fails (e.g., the input model
        isn't in the handler's globals).
        """
        try:
            sig = inspect.signature(self.handler, eval_str=True)
        except (NameError, SyntaxError):
            sig = inspect.signature(self.handler)
        params = list(sig.parameters.values())
        if len(params) == 1:
            ann = params[0].annotation
            if ann is self.input_model:
                return self.handler(parsed)
            if (
                ann is not inspect.Parameter.empty
                and inspect.isclass(ann)
                and issubclass(ann, BaseModel)
            ):
                return self.handler(parsed)
            # Last-resort: string-name match for unresolvable forward refs.
            if isinstance(ann, str) and ann == self.input_model.__name__:
                return self.handler(parsed)
        return self.handler(**parsed.model_dump())

    def _compute_idempotency_key(self, parsed: BaseModel) -> str:
        """Hash the inputs that define a logical action.

        Defaults to all fields. Handlers that include a ``session_id`` or
        cosmetic field (e.g., ``reason`` text) should pass
        ``idempotency_key_fields`` to scope the key to the action's identity,
        not the surrounding context.
        """
        if self.idempotency_key_fields:
            data = {k: getattr(parsed, k, None) for k in self.idempotency_key_fields}
        else:
            data = parsed.model_dump()
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def _record_audit(
        self,
        call_id: str,
        arguments: dict[str, Any],
        result: Any,
        error: ToolError | None = None,
        idempotent_hit: bool = False,
    ) -> None:
        if self.audit_log is None:
            return
        self.audit_log.record(
            tool_name=self.name,
            arguments=arguments,
            result=result,
            error=error,
            idempotent_hit=idempotent_hit,
            call_id=call_id,
        )

    def _validation_error_to_tool_error(self, e: ValidationError) -> ToolError:
        """Turn a Pydantic ValidationError into the tool's self-healing feedback.

        The agent reads the message, sees which field failed and how, and
        retries with a corrected payload — this is the cognitive-feedback
        contract that makes schema validation more than type checking.
        """
        problems: list[str] = []
        for raw in e.errors(include_url=False, include_input=False):
            loc = ".".join(str(p) for p in raw.get("loc", ())) or "<root>"
            problems.append(f"{loc}: {raw.get('msg', 'invalid')}")
        return ToolError(
            category=ErrorCategory.INVALID_INPUT,
            message="Invalid arguments. " + "; ".join(problems),
            retryable=True,
            suggested_action=(
                "Re-issue the call with arguments matching the schema. "
                "Each problem above names the field and what's wrong."
            ),
            details={"problems": problems},
        )


def tool(
    *,
    input_model: type[BaseModel],
    name: str | None = None,
    description: str | None = None,
    when_not_to_use: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    idempotent: bool = False,
    idempotency_key_fields: list[str] | tuple[str, ...] | None = None,
    idempotency_cache: IdempotencyCache | None = None,
    audit_log: AuditLog | None = None,
    output_filter: Callable[[Any], Any] | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator: wrap a handler function as a ``Tool``.

    Example::

        class LookupOrderInput(BaseModel):
            order_id: str = Field(description="Order number, format ORD-XXXXX.")

        @tool(
            name="lookup_order",
            description="Look up an order by order number.",
            when_not_to_use="Don't call without a confirmed order number.",
            input_model=LookupOrderInput,
        )
        def lookup_order(order_id: str) -> dict:
            return ORDERS[order_id]

    The decorator consumes the function's docstring as the description
    when one isn't passed explicitly, so handlers that already document
    themselves get usable schemas with no additional text.
    """

    def decorator(fn: Callable[..., Any]) -> Tool:
        return Tool(
            handler=fn,
            name=name or fn.__name__,
            description=description or (inspect.getdoc(fn) or fn.__name__),
            input_model=input_model,
            when_not_to_use=when_not_to_use,
            tags=tags,
            idempotent=idempotent,
            idempotency_key_fields=idempotency_key_fields,
            idempotency_cache=idempotency_cache,
            audit_log=audit_log,
            output_filter=output_filter,
        )

    return decorator


def refuse(reason: str, **details: Any) -> dict[str, Any]:
    """Build a structured refusal envelope.

    Use inside a tool handler when the right answer is to decline:

        return refuse("I am not authorized to prescribe medication.")

    The agent loop can detect refusals programmatically (presence of the
    ``"refusal"`` key) instead of trying to parse intent from a free-form
    string, which is more robust than relying on natural-language signals.
    """
    out: dict[str, Any] = {"refusal": reason}
    if details:
        out["details"] = details
    return out
