"""Drives one ADK invocation of a run and records how it ended.

A run moves through running -> (awaiting_approval -> running)* -> completed | failed | limit_exceeded.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import structlog
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService
from google.genai import types

from harness.adapters.repository import ApprovalRepository, RunRepository, TraceRepository
from harness.constants import ADK_CONFIRMATION_CALL, APP_NAME, StateKey, TerminationReason, TraceEventType
from harness.domain.model import RunStatus
from harness.observability import Tracer

log = structlog.get_logger("harness.core.orchestrator")


@dataclass(frozen=True)
class RunLimits:
    max_llm_calls: int
    # Hard backstop: the guardrail plugin stops a run gracefully at its time budget; this
    # cancels the invocation if a single model/tool call hangs past it.
    hard_timeout_s: float


@dataclass
class _Outcome:
    final_text: str | None = None
    error: str | None = None
    approvals: list[str] = field(default_factory=list)


class RunOrchestrator:
    def __init__(
        self,
        runner: Runner,
        sessions: BaseSessionService,
        runs: RunRepository,
        approvals: ApprovalRepository,
        traces: TraceRepository,
        tracer: Tracer,
        limits: RunLimits,
    ):
        self._runner = runner
        self._sessions = sessions
        self._runs = runs
        self._approvals = approvals
        self._traces = traces
        self._tracer = tracer
        self._limits = limits

    async def drive(self, run_id: str, user_id: str, message: types.Content) -> None:
        """Runs the agent until it finishes, pauses for approval, or a limit trips. Callers serialize per run."""
        outcome = _Outcome()
        reason: str | None = None
        try:
            async with asyncio.timeout(self._limits.hard_timeout_s):
                async for event in self._runner.run_async(
                    user_id=user_id,
                    session_id=run_id,
                    new_message=message,
                    run_config=RunConfig(max_llm_calls=self._limits.max_llm_calls),
                ):
                    await self._observe(run_id, event, outcome)
        except LlmCallsLimitExceededError as e:
            status, reason, outcome.error = RunStatus.LIMIT_EXCEEDED, TerminationReason.MAX_LLM_CALLS, str(e)
        except TimeoutError:
            status, reason = RunStatus.LIMIT_EXCEEDED, TerminationReason.HARD_TIMEOUT
            outcome.error = f"run exceeded {self._limits.hard_timeout_s}s"
        except Exception as e:  # any crash must end in a recorded FAILED run
            log.exception("run_crashed", run_id=run_id)
            status, outcome.error = RunStatus.FAILED, f"{type(e).__name__}: {e}"
        else:
            session = await self._sessions.get_session(app_name=APP_NAME, user_id=user_id, session_id=run_id)
            reason = session.state.get(StateKey.TERMINATION) if session else None
            status = self._resolve_status(outcome, reason)

        if reason:
            await self._tracer.emit(run_id, TraceEventType.LIMIT_EXCEEDED, name=reason, detail=outcome.error)
        await self._record(run_id, status, outcome, reason)

    @staticmethod
    def _resolve_status(outcome: _Outcome, termination_reason: str | None) -> RunStatus:
        if outcome.approvals:
            return RunStatus.AWAITING_APPROVAL
        if termination_reason:
            return RunStatus.LIMIT_EXCEEDED
        if outcome.error:
            return RunStatus.FAILED
        if outcome.final_text:
            return RunStatus.COMPLETED
        outcome.error = "model ended the turn without an answer"
        return RunStatus.FAILED

    async def _record(self, run_id: str, status: RunStatus, outcome: _Outcome, reason: str | None) -> None:
        trace = await self._traces.list_for_run(run_id)
        await self._runs.update(
            run_id,
            status=status,
            final_answer=outcome.final_text,
            error=outcome.error,
            termination_reason=reason,
            llm_calls=sum(1 for t in trace if t.type == TraceEventType.MODEL_CALL),
            tool_calls=sum(1 for t in trace if t.type == TraceEventType.TOOL_CALL and t.payload.get("executed")),
        )
        await self._tracer.emit(run_id, TraceEventType.RUN_STATUS, name=status, reason=reason, error=outcome.error)

    async def _observe(self, run_id: str, event: Event, outcome: _Outcome) -> None:
        for fc in event.get_function_calls():
            if fc.name == ADK_CONFIRMATION_CALL:
                original = (fc.args or {}).get("originalFunctionCall", {})
                await self._approvals.create(fc.id, run_id, original.get("name", "?"), original.get("args", {}))
                await self._tracer.emit(
                    run_id,
                    TraceEventType.APPROVAL_REQUESTED,
                    name=original.get("name"),
                    approval_id=fc.id,
                    args=original.get("args"),
                )
                outcome.approvals.append(fc.id)
        if event.error_code:
            outcome.error = f"{event.error_code}: {event.error_message}"
        if event.is_final_response() and event.content and event.content.parts:
            texts = [p.text for p in event.content.parts if p.text and not p.thought]
            if texts:
                outcome.final_text = "\n".join(texts)
                outcome.error = None
