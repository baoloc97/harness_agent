"""One repository per aggregate: runs, approvals, trace events."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.adapters.orm import Approval, Run, TraceEvent, utcnow
from harness.domain.model import ApprovalStatus, RunStatus

Sessions = async_sessionmaker[AsyncSession]


class RunRepository:
    def __init__(self, sessions: Sessions):
        self._sessions = sessions

    async def create(self, run_id: str, user_id: str, objective: str) -> Run:
        run = Run(id=run_id, user_id=user_id, objective=objective, status=RunStatus.RUNNING)
        async with self._sessions.begin() as s:
            s.add(run)
        return run

    async def get(self, run_id: str) -> Run | None:
        async with self._sessions() as s:
            return await s.get(Run, run_id)

    async def list_recent(self, limit: int = 50) -> list[Run]:
        async with self._sessions() as s:
            return list(await s.scalars(select(Run).order_by(Run.created_at.desc()).limit(limit)))

    async def update(self, run_id: str, **fields: Any) -> None:
        async with self._sessions.begin() as s:
            await s.execute(update(Run).where(Run.id == run_id).values(**fields, updated_at=utcnow()))


class ApprovalRepository:
    def __init__(self, sessions: Sessions):
        self._sessions = sessions

    async def create(self, approval_id: str, run_id: str, tool_name: str, tool_args: dict) -> None:
        async with self._sessions.begin() as s:
            s.add(
                Approval(
                    id=approval_id,
                    run_id=run_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    status=ApprovalStatus.PENDING,
                )
            )

    async def list_for_run(self, run_id: str) -> list[Approval]:
        async with self._sessions() as s:
            q = select(Approval).where(Approval.run_id == run_id).order_by(Approval.requested_at)
            return list(await s.scalars(q))

    async def decide(self, approval_id: str, approved: bool, decided_by: str, reason: str | None) -> bool:
        """Atomically moves a pending approval to a decision. False if it was not pending."""
        new_status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        async with self._sessions.begin() as s:
            result = await s.execute(
                update(Approval)
                .where(Approval.id == approval_id, Approval.status == ApprovalStatus.PENDING)
                .values(status=new_status, decided_by=decided_by, reason=reason, decided_at=utcnow())
            )
            return result.rowcount == 1

    async def mark_consumed(self, run_id: str) -> None:
        async with self._sessions.begin() as s:
            await s.execute(
                update(Approval)
                .where(Approval.run_id == run_id, Approval.status != ApprovalStatus.PENDING)
                .values(consumed=True)
            )


class TraceRepository:
    """Implements observability.TraceSink."""

    def __init__(self, sessions: Sessions):
        self._sessions = sessions

    async def add(
        self,
        run_id: str,
        type: str,
        name: str | None,
        invocation_id: str | None,
        latency_ms: float | None,
        payload: dict[str, Any],
    ) -> None:
        async with self._sessions.begin() as s:
            s.add(
                TraceEvent(
                    run_id=run_id,
                    type=type,
                    name=name,
                    invocation_id=invocation_id,
                    latency_ms=latency_ms,
                    payload=payload,
                )
            )

    async def list_for_run(self, run_id: str) -> list[TraceEvent]:
        async with self._sessions() as s:
            return list(await s.scalars(select(TraceEvent).where(TraceEvent.run_id == run_id).order_by(TraceEvent.id)))
