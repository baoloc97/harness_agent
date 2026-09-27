"""Use cases of the harness: start a run, decide an approval, read runs, traces and incidents.

One run == one ADK session (session id == run id). Execution itself is delegated to RunOrchestrator.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from google.adk.sessions import BaseSessionService
from google.genai import types

from harness.adapters.incidents import IncidentGateway
from harness.adapters.repository import ApprovalRepository, RunRepository, TraceRepository
from harness.constants import ADK_CONFIRMATION_CALL, APP_NAME, RUN_ID_PREFIX, TraceEventType
from harness.domain.errors import ConflictError, InvalidInputError, NotFoundError
from harness.domain.model import ApprovalStatus, RunStatus, new_id
from harness.observability import Tracer
from harness.service_layer.orchestrator import RunOrchestrator
from harness.views import IncidentView, RunSummary, RunView, TraceEventView


class RunLocks:
    """Serializes work on the same run within this process; entries are dropped when unused."""

    def __init__(self) -> None:
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def hold(self, run_id: str) -> AsyncIterator[None]:
        lock, users = self._locks.get(run_id, (asyncio.Lock(), 0))
        self._locks[run_id] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            lock, users = self._locks[run_id]
            if users == 1:
                del self._locks[run_id]
            else:
                self._locks[run_id] = (lock, users - 1)


class HarnessService:
    def __init__(
        self,
        orchestrator: RunOrchestrator,
        sessions: BaseSessionService,
        runs: RunRepository,
        approvals: ApprovalRepository,
        traces: TraceRepository,
        incidents: IncidentGateway,
        tracer: Tracer,
    ):
        self._orchestrator = orchestrator
        self._sessions = sessions
        self._runs = runs
        self._approvals = approvals
        self._traces = traces
        self._incidents = incidents
        self._tracer = tracer
        self._locks = RunLocks()

    async def start_run(self, objective: str, user_id: str = "default") -> RunView:
        objective = objective.strip()
        if not objective:
            raise InvalidInputError("objective must not be empty")
        run_id = new_id(RUN_ID_PREFIX)
        await self._runs.create(run_id, user_id, objective)
        await self._sessions.create_session(app_name=APP_NAME, user_id=user_id, session_id=run_id)
        await self._tracer.emit(run_id, TraceEventType.RUN_STARTED, objective=objective, user_id=user_id)
        async with self._locks.hold(run_id):
            await self._orchestrator.drive(
                run_id, user_id, types.Content(role="user", parts=[types.Part(text=objective)])
            )
        return await self.get_run(run_id)

    async def decide_approval(
        self, run_id: str, approval_id: str, approve: bool, decided_by: str = "operator", reason: str | None = None
    ) -> RunView:
        async with self._locks.hold(run_id):
            run = await self._runs.get(run_id)
            if run is None:
                raise NotFoundError(f"run {run_id} not found")
            if run.status != RunStatus.AWAITING_APPROVAL:
                raise ConflictError(f"run {run_id} is {run.status}, not awaiting approval")
            if approval_id not in {a.id for a in await self._approvals.list_for_run(run_id)}:
                raise NotFoundError(f"approval {approval_id} not found on run {run_id}")
            if not await self._approvals.decide(approval_id, approve, decided_by, reason):
                raise ConflictError(f"approval {approval_id} was already decided")
            await self._tracer.emit(
                run_id,
                TraceEventType.APPROVAL_DECIDED,
                name=approval_id,
                approved=approve,
                decided_by=decided_by,
                reason=reason,
            )

            approvals = await self._approvals.list_for_run(run_id)
            if any(a.status == ApprovalStatus.PENDING for a in approvals):
                return await self.get_run(run_id)  # wait until every parallel request is decided

            # Answer every confirmation request of this pause in a single message, as ADK expects.
            # The payload carries the decision details to ApprovalFeedbackPlugin.
            parts = [
                types.Part(
                    function_response=types.FunctionResponse(
                        id=a.id,
                        name=ADK_CONFIRMATION_CALL,
                        response={
                            "confirmed": a.status == ApprovalStatus.APPROVED,
                            "payload": {"decided_by": a.decided_by, "reason": a.reason},
                        },
                    )
                )
                for a in approvals
                if not a.consumed
            ]
            await self._approvals.mark_consumed(run_id)
            await self._runs.update(run_id, status=RunStatus.RUNNING)
            await self._orchestrator.drive(run_id, run.user_id, types.Content(role="user", parts=parts))
        return await self.get_run(run_id)

    async def get_run(self, run_id: str) -> RunView:
        run = await self._runs.get(run_id)
        if run is None:
            raise NotFoundError(f"run {run_id} not found")
        session = await self._sessions.get_session(app_name=APP_NAME, user_id=run.user_id, session_id=run_id)
        return RunView.build(run, await self._approvals.list_for_run(run_id), session.events if session else [])

    async def list_runs(self, limit: int = 50) -> list[RunSummary]:
        return [RunSummary.of(r) for r in await self._runs.list_recent(limit)]

    async def get_trace(self, run_id: str) -> list[TraceEventView]:
        if await self._runs.get(run_id) is None:
            raise NotFoundError(f"run {run_id} not found")
        return [TraceEventView.of(t) for t in await self._traces.list_for_run(run_id)]

    async def list_incidents(self) -> list[IncidentView]:
        return [IncidentView.of(i) for i in await self._incidents.list()]
