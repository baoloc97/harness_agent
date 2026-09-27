from fastapi.testclient import TestClient

from harness.adapters.llm import RuleBasedLlm
from harness.agent.tools.resilience import FaultInjector
from harness.bootstrap import build_harness
from harness.entrypoints.api import create_app
from tests.conftest import settings_for


def client_for(database_url: str) -> TestClient:
    harness = build_harness(settings_for(database_url), llm=RuleBasedLlm(), faults=FaultInjector())
    return TestClient(create_app(harness))


def test_api_full_flow_with_approval(database_url):
    with client_for(database_url) as client:
        created = client.post("/runs", json={"objective": "payment-api is slow, open an incident if needed"})
        assert created.status_code == 201
        run = created.json()
        assert run["status"] == "awaiting_approval"
        approval_id = run["pending_approvals"][0]["approval_id"]

        decided = client.post(
            f"/runs/{run['run_id']}/approvals/{approval_id}",
            json={"decision": "approve", "decided_by": "alice"},
        )
        assert decided.status_code == 200
        assert decided.json()["status"] == "completed"

        again = client.post(f"/runs/{run['run_id']}/approvals/{approval_id}", json={"decision": "approve"})
        assert again.status_code == 409

        trace = client.get(f"/runs/{run['run_id']}/trace").json()
        assert {"run_started", "model_call", "tool_call", "approval_requested", "approval_decided"} <= {
            t["type"] for t in trace
        }
        assert len(client.get("/incidents").json()) == 1


def test_api_validation_and_not_found(database_url):
    with client_for(database_url) as client:
        assert client.post("/runs", json={"objective": ""}).status_code == 422
        assert client.get("/runs/run_missing").status_code == 404
        bad = client.post("/runs/run_missing/approvals/x", json={"decision": "maybe"})
        assert bad.status_code == 422


def test_api_fault_injection_endpoint(database_url):
    with client_for(database_url) as client:
        assert (
            client.post("/debug/faults", json={"tool": "get_service_status", "faults": ["transient"]}).status_code
            == 202
        )
        run = client.post("/runs", json={"objective": "auth-service health check"}).json()
        assert run["status"] == "completed"
        tool_events = [
            t
            for t in client.get(f"/runs/{run['run_id']}/trace").json()
            if t["name"] == "get_service_status" and t["type"] == "tool_call"
        ]
        assert tool_events[0]["payload"]["attempts"] == 2
