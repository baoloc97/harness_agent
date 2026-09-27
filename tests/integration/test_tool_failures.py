from harness.adapters.llm import call, text
from harness.agent.tools.resilience import FaultInjector
from tests.conftest import tool_results, trace_of


async def test_transient_error_is_retried_transparently(make_service):
    faults = FaultInjector()
    faults.add("get_service_status", "transient")
    svc = await make_service([call("get_service_status", service_name="payment-api"), text("degraded")], faults=faults)

    run = await svc.start_run("status of payment-api")

    assert run.status == "completed"
    # the model never saw the failure
    assert tool_results(run, "get_service_status")[0]["status"] == "degraded"
    tool_event = trace_of(await svc.get_trace(run.run_id), "tool_call")[0]
    assert tool_event.payload["attempts"] == 2


async def test_timeouts_exhaust_retries_and_error_is_reported_to_model(make_service):
    faults = FaultInjector()
    faults.add("get_service_status", "timeout", "timeout", "timeout")
    seen = {}

    def final(req):
        seen["last"] = req.contents[-1].parts[0].function_response.response
        return text("status unavailable, try again later")

    svc = await make_service([call("get_service_status", service_name="payment-api"), final], faults=faults)
    run = await svc.start_run("status of payment-api")

    assert run.status == "completed"
    assert "ToolRetriesExhausted" in seen["last"]["error_type"]
    assert "timed out" in seen["last"]["error_details"]
    trace = await svc.get_trace(run.run_id)
    error_event = trace_of(trace, "tool_error")[0]
    assert error_event.payload["attempts"] == 3


async def test_permanent_error_is_not_retried(make_service):
    svc = await make_service([call("get_service_status", service_name="billing-api"), text("unknown service")])

    run = await svc.start_run("status of billing-api")

    assert run.status == "completed"
    error_event = trace_of(await svc.get_trace(run.run_id), "tool_error")[0]
    assert error_event.payload["attempts"] == 1
    assert "unknown service" in error_event.payload["error"]


async def test_malformed_tool_output_is_replaced_before_reaching_model(make_service):
    faults = FaultInjector()
    faults.add("get_service_status", "malformed")
    svc = await make_service(
        [call("get_service_status", service_name="payment-api"), text("could not verify")], faults=faults
    )

    run = await svc.start_run("status of payment-api")

    result = tool_results(run, "get_service_status")[0]
    assert result["error_type"] == "INVALID_TOOL_OUTPUT"
    assert "on fire" not in str(result)
    assert trace_of(await svc.get_trace(run.run_id), "tool_output_invalid")
