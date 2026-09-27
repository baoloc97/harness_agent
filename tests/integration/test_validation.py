import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from harness.adapters.llm import call, malformed, text
from harness.domain.errors import InvalidInputError
from tests.conftest import tool_results, trace_of


async def test_invalid_arguments_are_rejected_before_approval(make_service):
    svc = await make_service(
        [
            call(
                "create_incident",
                title="payment-api down",
                description="checkout failing for all users",
                severity="urgent",
            ),
            text("could not create incident"),
        ]
    )

    run = await svc.start_run("open an incident for payment-api")

    assert run.status == "completed"
    assert run.approvals == []  # the human was never asked
    result = tool_results(run, "create_incident")[0]
    assert result["error_type"] == "INVALID_ARGUMENTS"
    assert any("severity" in d for d in result["details"])
    assert await svc.list_incidents() == []


async def test_unexpected_extra_argument_is_rejected(make_service):
    svc = await make_service([call("get_service_status", service_name="payment-api", region="eu"), text("ok")])
    run = await svc.start_run("status")
    assert tool_results(run, "get_service_status")[0]["error_type"] == "INVALID_ARGUMENTS"


async def test_unknown_tool_is_answered_without_crashing(make_service):
    svc = await make_service([call("restart_service", service_name="payment-api"), text("I cannot restart services.")])

    run = await svc.start_run("restart payment-api")

    assert run.status == "completed"
    assert "not found" in str(tool_results(run, "restart_service")[0]).lower()


async def test_malformed_llm_response_is_recovered(make_service):
    svc = await make_service([malformed(), call("get_service_status", service_name="payment-api"), text("degraded")])

    run = await svc.start_run("status of payment-api")

    assert run.status == "completed"
    assert run.final_answer == "degraded"
    model_calls = trace_of(await svc.get_trace(run.run_id), "model_call")
    assert model_calls[0].payload["error_code"] == "MALFORMED_FUNCTION_CALL"


async def test_persistently_malformed_llm_fails_the_run(make_service):
    svc = await make_service([malformed()] * 5, max_model_retries=2)

    run = await svc.start_run("status of payment-api")

    assert run.status == "failed"
    assert "MALFORMED_FUNCTION_CALL" in run.error


async def test_empty_objective_is_rejected(make_service):
    svc = await make_service([])
    with pytest.raises(InvalidInputError):
        await svc.start_run("   ")


async def test_empty_llm_response_fails_the_run(make_service):
    svc = await make_service([LlmResponse(content=types.Content(role="model", parts=[]))])

    run = await svc.start_run("status of payment-api")

    assert run.status == "failed"
    assert "without an answer" in run.error
