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
from harness.constants import ADK_CONFIRMATION_CALL, TraceEventType
from harness.domain.model import RunStatus
from harness.observability import configure_logging
from harness.service_layer.service import HarnessService
from harness.views import HistoryItem, RunView, TraceEventView

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


def _trace_details(e: TraceEventView) -> str:
    p = e.payload
    if e.type == TraceEventType.MODEL_CALL:
        calls = ", ".join(f["name"] for f in p.get("function_calls") or [])
        tokens = f"  tokens {p['prompt_tokens']}+{p['output_tokens']}" if p.get("prompt_tokens") else ""
        return (f"calls {calls}" if calls else "final answer") + tokens
    if e.type == TraceEventType.TOOL_CALL:
        if not p.get("executed"):
            result = p.get("result") or {}
            if "requires confirmation" in str(result.get("error", "")):
                return "paused for approval"
            return f"not executed: {result.get('error_type') or result.get('error', '')}"
        approved = f"  approved={p['approved']}" if p.get("approved") is not None else ""
        return f"status={p.get('status')}  attempts={p.get('attempts')}{approved}"
    if e.type == TraceEventType.APPROVAL_REQUESTED:
        return "waiting for a human"
    if e.type == TraceEventType.APPROVAL_DECIDED:
        reason = f"  reason: {p['reason']}" if p.get("reason") else ""
        return f"approved={p.get('approved')} by {p.get('decided_by')}{reason}"
    if e.type in (TraceEventType.TOOL_ERROR, TraceEventType.LIMIT_EXCEEDED, TraceEventType.RUN_STATUS):
        return str(p.get("error") or p.get("detail") or "")
    return ""


def _print_trace_table(events: list[TraceEventView]) -> None:
    typer.echo(f"{'time':8}  {'event':19} {'name':22} {'ms':>6}  details")
    typer.echo("-" * 90)
    tools_by_approval = {
        e.payload.get("approval_id"): e.name for e in events if e.type == TraceEventType.APPROVAL_REQUESTED
    }
    for e in events:
        if e.type in (TraceEventType.INVOCATION_START, TraceEventType.INVOCATION_END):
            continue
        ms = f"{e.latency_ms:.0f}" if e.latency_ms is not None else ""
        # approval events are named by approval id; show the tool the approval is about
        name = tools_by_approval.get(e.name or "", e.name or "")
        typer.echo(f"{e.ts:%H:%M:%S}  {e.type:19} {name[:22]:22} {ms:>6}  {_trace_details(e)}")


@app.command()
def trace(
    run_id: str, table: bool = typer.Option(False, "--table", help="Readable table instead of JSON lines")
) -> None:
    """Print the execution trace of a run (JSON lines by default)."""

    async def work(svc: HarnessService) -> None:
        events = await svc.get_trace(run_id)
        if table:
            _print_trace_table(events)
        else:
            for event in events:
                typer.echo(event.model_dump_json())

    _run_with_service(work)


if __name__ == "__main__":
    app()
