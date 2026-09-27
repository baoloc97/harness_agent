from harness.adapters.llm import RuleBasedLlm, call, text
from tests.conftest import tool_results, trace_of


async def test_happy_path_completes_with_trace_and_history(make_service):
    svc = await make_service(
        [
            call("search_knowledge_base", query="auth-service login failures"),
            call("get_service_status", service_name="auth-service"),
            text("auth-service is healthy; see KB-003. No incident needed."),
        ]
    )

    run = await svc.start_run("Users report login failures on auth-service")

    assert run.status == "completed"
    assert run.final_answer.startswith("auth-service is healthy")
    assert run.llm_calls == 3 and run.tool_calls == 2
    assert tool_results(run, "get_service_status")[0]["status"] == "healthy"
    assert tool_results(run, "search_knowledge_base")[0]["results"][0]["id"] == "KB-003"

    trace = await svc.get_trace(run.run_id)
    assert [t.type for t in trace][0] == "run_started"
    tool_events = trace_of(trace, "tool_call")
    assert [t.name for t in tool_events] == ["search_knowledge_base", "get_service_status"]
    assert all(t.latency_ms is not None and t.payload["attempts"] == 1 for t in tool_events)
    assert trace_of(trace, "run_status")[-1].name == "completed"


async def test_rule_based_planner_end_to_end_with_approval(make_service):
    svc = await make_service(llm=RuleBasedLlm())

    run = await svc.start_run("inventory-db is failing, open an incident if needed")
    assert run.status == "awaiting_approval"
    pending = run.pending_approvals[0]
    assert pending.tool == "create_incident" and pending.args["severity"] == "critical"

    run = await svc.decide_approval(run.run_id, pending.approval_id, approve=True)
    assert run.status == "completed"
    assert "INC-" in run.final_answer


async def test_run_state_is_queryable_after_completion(make_service):
    svc = await make_service([text("nothing to do")])
    run = await svc.start_run("hello")
    again = await svc.get_run(run.run_id)
    assert again.status == "completed"
    assert [r.run_id for r in await svc.list_runs()] == [run.run_id]


async def test_kb_top_k_setting_limits_search_results(make_service):
    svc = await make_service(
        [call("search_knowledge_base", query="payment-api latency incident severity"), text("ok")], kb_top_k=1
    )

    run = await svc.start_run("search")

    assert len(tool_results(run, "search_knowledge_base")[0]["results"]) == 1
