"""Read models returned by the service and serialized by the API (also used as FastAPI response models)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from google.adk.events import Event
from pydantic import BaseModel, computed_field

from harness.adapters.incidents import Incident
from harness.adapters.orm import Approval, Run, TraceEvent
from harness.domain.model import ApprovalStatus


class ApprovalView(BaseModel):
    approval_id: str
    tool: str
    args: dict[str, Any]
    status: ApprovalStatus
    decided_by: str | None
    reason: str | None
    requested_at: datetime | None
    decided_at: datetime | None

    @classmethod
    def of(cls, a: Approval) -> ApprovalView:
        return cls(
            approval_id=a.id,
            tool=a.tool_name,
            args=a.tool_args,
            status=a.status,
            decided_by=a.decided_by,
            reason=a.reason,
            requested_at=a.requested_at,
            decided_at=a.decided_at,
        )


class HistoryItem(BaseModel):
    author: str
    type: Literal["text", "tool_call", "tool_result"]
    text: str | None = None
    tool: str | None = None
    args: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


class RunSummary(BaseModel):
    run_id: str
    status: str
    objective: str
    created_at: datetime

    @classmethod
    def of(cls, r: Run) -> RunSummary:
        return cls(run_id=r.id, status=r.status, objective=r.objective, created_at=r.created_at)


class RunView(RunSummary):
    user_id: str
    final_answer: str | None
    error: str | None
    termination_reason: str | None
    llm_calls: int
    tool_calls: int
    updated_at: datetime
    approvals: list[ApprovalView]
    history: list[HistoryItem]

    @computed_field
    @property
    def pending_approvals(self) -> list[ApprovalView]:
        return [a for a in self.approvals if a.status == ApprovalStatus.PENDING]

    @classmethod
    def build(cls, run: Run, approvals: list[Approval], events: list[Event]) -> RunView:
        return cls(
            run_id=run.id,
            status=run.status,
            objective=run.objective,
            created_at=run.created_at,
            user_id=run.user_id,
            final_answer=run.final_answer,
            error=run.error,
            termination_reason=run.termination_reason,
            llm_calls=run.llm_calls,
            tool_calls=run.tool_calls,
            updated_at=run.updated_at,
            approvals=[ApprovalView.of(a) for a in approvals],
            history=history_from_events(events),
        )


class TraceEventView(BaseModel):
    ts: datetime
    type: str
    name: str | None
    invocation_id: str | None
    latency_ms: float | None
    payload: dict[str, Any]

    @classmethod
    def of(cls, t: TraceEvent) -> TraceEventView:
        return cls(
            ts=t.ts, type=t.type, name=t.name, invocation_id=t.invocation_id, latency_ms=t.latency_ms, payload=t.payload
        )


class IncidentView(BaseModel):
    incident_id: str
    title: str
    description: str
    severity: str
    created_at: datetime

    @classmethod
    def of(cls, i: Incident) -> IncidentView:
        return cls(
            incident_id=i.id, title=i.title, description=i.description, severity=i.severity, created_at=i.created_at
        )


def history_from_events(events: list[Event]) -> list[HistoryItem]:
    """Compact view of the ADK conversation."""
    items: list[HistoryItem] = []
    for e in events:
        for p in e.content.parts if e.content and e.content.parts else []:
            if p.text and not p.thought:
                items.append(HistoryItem(author=e.author, type="text", text=p.text))
            elif p.function_call:
                items.append(
                    HistoryItem(author=e.author, type="tool_call", tool=p.function_call.name, args=p.function_call.args)
                )
            elif p.function_response:
                items.append(
                    HistoryItem(
                        author=e.author,
                        type="tool_result",
                        tool=p.function_response.name,
                        result=p.function_response.response,
                    )
                )
    return items
