import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from harness.adapters.llm import call, text
from harness.domain.errors import ConflictError, NotFoundError
from tests.conftest import tool_results, unique_title

INCIDENT_ARGS = dict(description="checkout p95 1450ms, 3.2% errors", severity="high")


async def _paused_run(svc, title):
    before = len(await svc.list_incidents())
    run = await svc.start_run("payment-api is slow, open an incident")
    assert run.status == "awaiting_approval"
    assert len(await svc.list_incidents()) == before  # nothing happens before approval
    return run, run.pending_approvals[0]


async def test_approved_call_executes_and_run_completes(make_service):
    title = unique_title()
    svc = await make_service([call("create_incident", title=title, **INCIDENT_ARGS), text("Incident opened.")])
    run, pending = await _paused_run(svc, title)
    assert pending.args["title"] == title

    run = await svc.decide_approval(run.run_id, pending.approval_id, approve=True, decided_by="alice")

    assert run.status == "completed"
    assert run.approvals[0].status == "approved" and run.approvals[0].decided_by == "alice"
    incidents = await svc.list_incidents()
    assert len(incidents) == 1 and incidents[0].title == title
    assert tool_results(run, "create_incident")[-1]["incident_id"] == incidents[0].incident_id


async def test_rejected_call_is_not_executed_and_model_is_told(make_service):
    seen = {}

    def final(req):
        seen["response"] = req.contents[-1].parts[0].function_response.response
        return text("Incident creation was rejected by the operator.")

    svc = await make_service([call("create_incident", title=unique_title(), **INCIDENT_ARGS), final])
    run, pending = await _paused_run(svc, None)

    run = await svc.decide_approval(run.run_id, pending.approval_id, approve=False, reason="duplicate of INC-1")

    assert run.status == "completed"
    assert run.approvals[0].status == "rejected"
    assert seen["response"]["error_type"] == "REJECTED_BY_OPERATOR"
    assert seen["response"]["reason"] == "duplicate of INC-1"
    assert seen["response"]["rejected_by"] == "operator"
    assert await svc.list_incidents() == []


async def test_approval_cannot_be_decided_twice(make_service):
    svc = await make_service([call("create_incident", title=unique_title(), **INCIDENT_ARGS), text("done")])
    run, pending = await _paused_run(svc, None)
    await svc.decide_approval(run.run_id, pending.approval_id, approve=True)

    with pytest.raises(ConflictError):
        await svc.decide_approval(run.run_id, pending.approval_id, approve=True)


async def test_unknown_approval_id_is_rejected(make_service):
    svc = await make_service([call("create_incident", title=unique_title(), **INCIDENT_ARGS), text("done")])
    run, _ = await _paused_run(svc, None)
    with pytest.raises(NotFoundError):
        await svc.decide_approval(run.run_id, "adk-does-not-exist", approve=True)


async def test_paused_run_survives_a_process_restart(make_service):
    title = unique_title()
    steps = [call("create_incident", title=title, **INCIDENT_ARGS), text("Incident opened.")]
    first = await make_service(steps[:1])
    run, pending = await _paused_run(first, title)

    # A new service instance (new process) over the same database resumes the run.
    second = await make_service(steps[1:])
    run = await second.decide_approval(run.run_id, pending.approval_id, approve=True)

    assert run.status == "completed"
    assert len(await second.list_incidents()) == 1


async def test_repeated_incident_is_deduplicated(make_service):
    title = unique_title()
    svc = await make_service(
        [
            call("create_incident", title=title, **INCIDENT_ARGS),
            text("opened"),
            call("create_incident", title=title, **INCIDENT_ARGS),
            text("opened again"),
        ]
    )
    for _ in range(2):
        run, pending = await _paused_run(svc, title)
        run = await svc.decide_approval(run.run_id, pending.approval_id, approve=True)

    assert tool_results(run, "create_incident")[-1]["status"] == "duplicate"
    assert len(await svc.list_incidents()) == 1


async def test_counts_only_executed_tool_calls(make_service):
    svc = await make_service(
        [
            call("get_service_status", service_name="payment-api"),
            call("create_incident", title=unique_title(), **INCIDENT_ARGS),
            text("done"),
        ]
    )
    run = await svc.start_run("payment-api is slow, open an incident")
    assert run.tool_calls == 1  # the approval placeholder is not an execution
    run = await svc.decide_approval(run.run_id, run.pending_approvals[0].approval_id, approve=True)
    assert run.tool_calls == 2


async def test_parallel_calls_needing_approval_resume_only_when_all_decided(make_service):
    parallel = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="create_incident", args={"title": unique_title(), **INCIDENT_ARGS}
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        name="create_incident", args={"title": unique_title(), **INCIDENT_ARGS}
                    )
                ),
            ],
        )
    )
    svc = await make_service([parallel, text("one incident opened, one rejected")])

    run = await svc.start_run("open incidents")
    assert run.status == "awaiting_approval"
    first, second = run.pending_approvals

    run = await svc.decide_approval(run.run_id, first.approval_id, approve=True)
    assert run.status == "awaiting_approval"  # still waiting for the second decision
    assert await svc.list_incidents() == []

    run = await svc.decide_approval(run.run_id, second.approval_id, approve=False, reason="duplicate")
    assert run.status == "completed"
    incidents = await svc.list_incidents()
    assert [i.title for i in incidents] == [first.args["title"]]
