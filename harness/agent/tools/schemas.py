"""Output contracts for the tools. Inputs are declared by each tool's type-annotated signature."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KBHit(_Strict):
    id: str = Field(pattern=r"^KB-\d{3}$")
    title: str
    service: str | None
    snippet: str
    score: float = Field(ge=0)


class SearchKnowledgeBaseOutput(_Strict):
    query: str
    results: list[KBHit]


class ServiceStatusOutput(_Strict):
    service_name: str
    status: Literal["healthy", "degraded", "down"]
    latency_p95_ms: float | None
    error_rate_pct: float = Field(ge=0, le=100)
    last_deploy: str
    message: str


class CreateIncidentOutput(_Strict):
    incident_id: str = Field(pattern=r"^INC-[0-9A-F]{8}$")
    status: Literal["created", "duplicate"]
    title: str
    severity: Literal["low", "medium", "high", "critical"]
