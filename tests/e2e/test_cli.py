from typer.testing import CliRunner

from harness.adapters.llm import RuleBasedLlm
from harness.agent.tools.resilience import FaultInjector
from harness.bootstrap import build_harness
from harness.entrypoints import cli
from tests.conftest import settings_for

OBJECTIVE = "payment-api is slow, open an incident if needed"


def use_service(monkeypatch, database_url):
    monkeypatch.setattr(
        cli,
        "_harness",
        lambda: build_harness(settings_for(database_url), llm=RuleBasedLlm(), faults=FaultInjector()),
    )


def test_cli_prompts_and_approves(monkeypatch, database_url):
    use_service(monkeypatch, database_url)

    result = CliRunner().invoke(cli.app, ["run", OBJECTIVE], input="y\n")

    assert result.exit_code == 0, result.output
    assert "Approval required: create_incident" in result.output
    assert "status=completed" in result.output
    assert "Incident INC-" in result.output


def test_cli_reject_records_reason(monkeypatch, database_url):
    use_service(monkeypatch, database_url)

    result = CliRunner().invoke(cli.app, ["run", OBJECTIVE], input="n\nalready tracked\n")

    assert result.exit_code == 0, result.output
    assert "Incident not created" in result.output
    assert "Reason: already tracked" in result.output  # reached the agent, not just echoed input


def test_cli_show_and_trace(monkeypatch, database_url):
    use_service(monkeypatch, database_url)
    runner = CliRunner()
    runner.invoke(cli.app, ["run", OBJECTIVE, "--auto-approve"])
    run_id = next(
        line.split()[1]
        for line in runner.invoke(cli.app, ["run", "auth-service health check"]).output.splitlines()
        if line.startswith("run ")
    )

    shown = runner.invoke(cli.app, ["show", run_id])
    traced = runner.invoke(cli.app, ["trace", run_id])

    assert "status=completed" in shown.output
    assert '"type":"run_started"' in traced.output


def test_cli_trace_table_is_readable(monkeypatch, database_url):
    use_service(monkeypatch, database_url)
    runner = CliRunner()
    output = runner.invoke(cli.app, ["run", OBJECTIVE, "--auto-approve"]).output
    run_id = next(line.split()[1] for line in output.splitlines() if line.startswith("run "))

    table = runner.invoke(cli.app, ["trace", run_id, "--table"]).output

    assert "paused for approval" in table
    assert "approval_decided    create_incident" in table
    assert "approved=True by cli" in table
    assert "attempts=1  approved=True" in table
