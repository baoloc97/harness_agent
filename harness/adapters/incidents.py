"""The external incident management system, behind an interface.

Tools depend on IncidentGateway only. MockIncidentGateway stores incidents in a local table;
a real deployment would provide e.g. a PagerDuty or Jira implementation with the same shape.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy import DateTime, String, Text, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from harness.adapters.orm import Base, utcnow
from harness.adapters.repository import Sessions
from harness.constants import INCIDENT_ID_PREFIX


@dataclass(frozen=True)
class Incident:
    id: str
    title: str
    description: str
    severity: str
    created_at: datetime


class IncidentGateway(Protocol):
    async def create(self, title: str, description: str, severity: str, dedup_key: str) -> tuple[Incident, bool]:
        """Opens an incident. Returns (incident, created); an existing incident with the same dedup key is reused."""
        ...

    async def list(self) -> list[Incident]: ...


class _IncidentRow(Base):
    __tablename__ = "mock_incidents"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(16))
    dedup_key: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    def to_incident(self) -> Incident:
        return Incident(self.id, self.title, self.description, self.severity, self.created_at)


class MockIncidentGateway:
    def __init__(self, sessions: Sessions):
        self._sessions = sessions

    async def create(self, title: str, description: str, severity: str, dedup_key: str) -> tuple[Incident, bool]:
        existing = await self._find(dedup_key)
        if existing:
            return existing, False
        row = _IncidentRow(
            id=f"{INCIDENT_ID_PREFIX}-{uuid.uuid4().hex[:8].upper()}",
            title=title,
            description=description,
            severity=severity,
            dedup_key=dedup_key,
            created_at=utcnow(),
        )
        try:
            async with self._sessions.begin() as s:
                s.add(row)
        except IntegrityError:  # a concurrent call won the race on the unique dedup key
            return await self._find(dedup_key), False
        return row.to_incident(), True

    async def list(self) -> list[Incident]:
        async with self._sessions() as s:
            rows = await s.scalars(select(_IncidentRow).order_by(_IncidentRow.created_at))
            return [r.to_incident() for r in rows]

    async def _find(self, dedup_key: str) -> Incident | None:
        async with self._sessions() as s:
            row = await s.scalar(select(_IncidentRow).where(_IncidentRow.dedup_key == dedup_key))
            return row.to_incident() if row else None
