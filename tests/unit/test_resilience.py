import asyncio

import pytest

from harness.agent.tools.resilience import (
    FaultInjector,
    PermanentToolError,
    RetryPolicy,
    ToolRetriesExhausted,
    TransientToolError,
    resilient,
)

FAST = RetryPolicy(timeout_s=0.05, max_attempts=3, backoff_base_s=0.001)


def flaky(*errors: Exception):
    """A tool that raises the given errors in order, then succeeds."""
    calls = {"n": 0}

    async def tool(x: int) -> dict:
        calls["n"] += 1
        if errors[calls["n"] - 1 :]:
            raise errors[calls["n"] - 1]
        return {"x": x}

    return tool, calls


async def test_transient_errors_are_retried_until_success():
    tool, calls = flaky(TransientToolError("503"), TransientToolError("503"))
    assert await resilient(tool, FAST)(x=1) == {"x": 1}
    assert calls["n"] == 3


async def test_permanent_error_is_raised_immediately():
    tool, calls = flaky(PermanentToolError("400"))
    with pytest.raises(PermanentToolError):
        await resilient(tool, FAST)(x=1)
    assert calls["n"] == 1


async def test_timeouts_exhaust_the_policy():
    async def hangs() -> dict:
        await asyncio.sleep(1)
        return {}

    with pytest.raises(ToolRetriesExhausted) as err:
        await resilient(hangs, FAST)()
    assert err.value.attempts == 3


def test_wrapper_keeps_the_signature_adk_reads():
    import inspect

    tool, _ = flaky()
    assert list(inspect.signature(resilient(tool, FAST)).parameters) == ["x"]


async def test_fault_injector_parses_config_and_consumes_in_order():
    faults = FaultInjector.from_string("get_service_status=transient,malformed; create_incident=fatal")
    with pytest.raises(TransientToolError):
        await faults.apply("get_service_status")
    assert await faults.apply("get_service_status") == "malformed"
    assert await faults.apply("get_service_status") is None
    with pytest.raises(PermanentToolError):
        await faults.apply("create_incident")


def test_fault_injector_rejects_unknown_kinds():
    with pytest.raises(ValueError):
        FaultInjector.from_string("get_service_status=explode")
