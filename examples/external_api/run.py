"""Demo: defensive wrapping in action.

Drives the weather tool through a deterministic failure pattern so each
behavior — retry recovery, timeout conversion, circuit breaker tripping,
idempotent cache hit — is visible in the output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.external_api.fake_api import FakeWeatherAPI  # noqa: E402
from examples.external_api.tools import build_weather_tool  # noqa: E402


def divider(title: str) -> None:
    print(f"\n===== {title} =====")


def main() -> None:
    divider("Scenario 1: transient errors recovered by retries")
    api = FakeWeatherAPI(failure_pattern=["error", "error", "ok"])
    tool, deps = build_weather_tool(api=api)
    result = tool(city="Berlin")
    print(json.dumps(result, indent=2))
    print(f"upstream calls: {api.call_count}")
    print(f"breaker state : {deps['breaker'].state.value}")

    divider("Scenario 2: idempotency — second call hits cache, no upstream")
    before = api.call_count
    cached = tool(city="Berlin")
    print(json.dumps(cached, indent=2))
    print(f"upstream calls during second invocation: {api.call_count - before}")

    divider("Scenario 3: timeout converted to structured error")
    slow_api = FakeWeatherAPI(failure_pattern=["timeout"], slow_seconds=2.0)
    slow_tool, _ = build_weather_tool(api=slow_api, timeout_seconds=0.2)
    print(json.dumps(slow_tool(city="Tokyo"), indent=2))

    divider("Scenario 4: persistent failures trip the circuit breaker")
    bad_api = FakeWeatherAPI(failure_pattern=["error"])
    bad_tool, bad_deps = build_weather_tool(api=bad_api)
    for n in range(3):
        out = bad_tool(city=f"City{n}")
        print(f"call {n+1} → category={_category(out)} state={bad_deps['breaker'].state.value}")
    blocked = bad_tool(city="City99")
    print(json.dumps(blocked, indent=2))

    divider("Audit summary (last tool only)")
    print(json.dumps(bad_deps["audit_log"].metrics(), indent=2))


def _category(envelope: dict) -> str:
    if isinstance(envelope, dict) and "error" in envelope:
        return envelope["error"].get("category", "?")
    return "ok"


if __name__ == "__main__":
    main()
