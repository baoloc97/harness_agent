"""CLI. Examples:

harness run "payment-api is slow, investigate and open an incident if needed"
harness show <run_id>
harness trace <run_id>
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import typer

from harness.bootstrap import Harness, build_harness
from harness.config import get_settings
from harness.constants import ADK_CONFIRMATION_CALL
from harness.domain.model import RunStatus
from harness.observability import configure_logging
from harness.service_layer.service import HarnessService
from harness.views import HistoryItem, RunView

app = typer.Typer(add_completion=False, help="Ops agent harness")


def _harness() -> Harness:
    settings = get_settings()
    configure_logging(settings.log_level)
    return build_harness(settings)


def _is_confirmation_placeholder(h: HistoryItem) -> bool:
    # ADK answers a paused call with this placeholder until the human decides.
    return "requires confirmation" in str((h.result or {}).get("error", ""))


def _print_run(run: RunView) -> None:
    typer.secho(f"\nrun {run.run_id}  status={run.status}", bold=True)
    for h in run.history:
        if h.type == "tool_call" and h.tool != ADK_CONFIRMATION_CALL:
            typer.echo(f"  → {h.tool}({json.dumps(h.args, ensure_ascii=False)})")
        elif h.type == "tool_result" and h.tool != ADK_CONFIRMATION_CALL and not _is_confirmation_placeholder(h):
            typer.echo(f"  ← {h.tool}: {json.dumps(h.result, ensure_ascii=False)[:200]}")
    if run.final_answer:
        typer.secho("\n" + run.final_answer, fg=typer.colors.GREEN)
    if run.error:
        typer.secho(f"error: {run.error}", fg=typer.colors.RED)
    if run.termination_reason:
        typer.secho(f"stopped by harness: {run.termination_reason}", fg=typer.colors.YELLOW)
    typer.echo(f"llm_calls={run.llm_calls} tool_calls={run.tool_calls}")


def _run_with_service(work: Callable[[HarnessService], Awaitable[None]]) -> None:
    async def main() -> None:
        harness = _harness()
        await harness.start()
        try:
            await work(harness.service)
        finally:
            await harness.stop()

    asyncio.run(main())


@app.command()
def run(
    objective: str, user_id: str = "cli", auto_approve: bool = typer.Option(False, help="Approve without prompting")
) -> None:
    """Run an objective; prompts for approval before any incident is created."""

    async def work(svc: HarnessService) -> None:
        result = await svc.start_run(objective, user_id)
        while result.status == RunStatus.AWAITING_APPROVAL:
            for approval in result.pending_approvals:
                typer.secho(f"\nApproval required: {approval.tool}", fg=typer.colors.YELLOW, bold=True)
                typer.echo(json.dumps(approval.args, indent=2, ensure_ascii=False))
                approved = auto_approve or typer.confirm("Approve this call?", default=False)
                reason = None if approved else typer.prompt("Reason for rejecting", default="rejected by operator")
                result = await svc.decide_approval(result.run_id, approval.approval_id, approved, "cli", reason)
        _print_run(result)

    _run_with_service(work)


@app.command()
def show(run_id: str) -> None:
    """Show a run's status, history and final answer."""

    async def work(svc: HarnessService) -> None:
        _print_run(await svc.get_run(run_id))

    _run_with_service(work)


@app.command()
def trace(run_id: str) -> None:
    """Print the execution trace of a run as JSON lines."""

    async def work(svc: HarnessService) -> None:
        for event in await svc.get_trace(run_id):
            typer.echo(event.model_dump_json())

    _run_with_service(work)


if __name__ == "__main__":
    app()
