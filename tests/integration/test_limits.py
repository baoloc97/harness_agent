import itertools

from harness.adapters.llm import ScriptedLlm, call, text
from tests.conftest import trace_of


def endless_searches():
    counter = itertools.count()
    return [lambda req: call("search_knowledge_base", query=f"query {next(counter)}")] * 100


async def test_max_llm_calls_stops_the_run(make_service):
    svc = await make_service(endless_searches(), max_llm_calls=4, max_tool_calls=100)

    run = await svc.start_run("loop forever")

    assert run.status == "limit_exceeded"
    assert run.termination_reason == "max_llm_calls_exceeded"
    assert run.llm_calls == 4


async def test_max_tool_calls_stops_the_run_gracefully(make_service):
    svc = await make_service(endless_searches(), max_tool_calls=3, max_llm_calls=50)

    run = await svc.start_run("loop forever")

    assert run.status == "limit_exceeded"
    assert run.termination_reason == "max_tool_calls_exceeded"
    assert "[harness] Run stopped" in run.final_answer


async def test_identical_calls_are_blocked(make_service):
    same = [call("get_service_status", service_name="payment-api")] * 4
    svc = await make_service(same + [text("giving up")], max_identical_tool_calls=2)

    run = await svc.start_run("status of payment-api")

    assert run.status == "completed"
    trace = await svc.get_trace(run.run_id)
    executed = [t for t in trace_of(trace, "tool_call") if t.payload["status"] == "ok"]
    assert len(executed) == 2
    assert len(trace_of(trace, "loop_detected")) == 2


async def test_time_budget_stops_the_run(make_service):
    slow = ScriptedLlm(endless_searches(), delay_s=0.2)
    svc = await make_service(llm=slow, run_time_budget_s=0.5, max_llm_calls=100, max_tool_calls=100)

    run = await svc.start_run("slow loop")

    assert run.status == "limit_exceeded"
    assert run.termination_reason == "time_budget_exceeded"
    assert run.llm_calls <= 5


async def test_hung_model_call_is_cancelled_by_hard_timeout(make_service):
    # The guardrail can only stop the run between steps; a model call that hangs is cut by the backstop.
    hung = ScriptedLlm([text("never returned in time")], delay_s=30)
    svc = await make_service(llm=hung, run_time_budget_s=0.3, tool_timeout_s=0.1, tool_max_attempts=1)

    run = await svc.start_run("hang")

    assert run.status == "limit_exceeded"
    assert run.termination_reason == "hard_timeout"
