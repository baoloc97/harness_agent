import asyncio

import pytest

from harness.domain.model import RunStatus
from harness.service_layer.orchestrator import RunOrchestrator, _Outcome
from harness.service_layer.service import RunLocks


@pytest.mark.parametrize(
    ("outcome", "termination", "expected"),
    [
        (_Outcome(final_text="done", approvals=["a1"]), None, RunStatus.AWAITING_APPROVAL),
        (_Outcome(final_text="[harness] stopped"), "max_tool_calls_exceeded", RunStatus.LIMIT_EXCEEDED),
        (_Outcome(error="MALFORMED_FUNCTION_CALL: ..."), None, RunStatus.FAILED),
        (_Outcome(final_text="done"), None, RunStatus.COMPLETED),
        (_Outcome(), None, RunStatus.FAILED),
    ],
)
def test_run_status_resolution(outcome, termination, expected):
    assert RunOrchestrator._resolve_status(outcome, termination) == expected


def test_empty_outcome_explains_the_failure():
    outcome = _Outcome()
    RunOrchestrator._resolve_status(outcome, None)
    assert outcome.error == "model ended the turn without an answer"


async def test_run_locks_serialize_work_on_the_same_run_and_clean_up():
    locks, order = RunLocks(), []

    async def work(tag: str) -> None:
        async with locks.hold("run_1"):
            order.append(f"{tag}-start")
            await asyncio.sleep(0.01)
            order.append(f"{tag}-end")

    await asyncio.gather(work("a"), work("b"))

    assert order in (["a-start", "a-end", "b-start", "b-end"], ["b-start", "b-end", "a-start", "a-end"])
    assert locks._locks == {}
