"""Mock implementations of the three ops tools."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from google.adk.tools import ToolContext
from pydantic import Field

from harness.adapters.incidents import IncidentGateway
from harness.agent.tools.registry import ToolDefinition
from harness.agent.tools.resilience import FaultInjector, PermanentToolError
from harness.agent.tools.schemas import CreateIncidentOutput, SearchKnowledgeBaseOutput, ServiceStatusOutput
from harness.constants import ToolName

Severity = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True)
class MockDataset:
    knowledge_base: list[dict]
    services: dict[str, dict]

    @classmethod
    def load(cls, data_dir: Path) -> MockDataset:
        return cls(
            knowledge_base=json.loads((data_dir / "knowledge_base.json").read_text()),
            services=json.loads((data_dir / "services.json").read_text()),
        )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def mock_tool_definitions(
    dataset: MockDataset, incidents: IncidentGateway, faults: FaultInjector, *, kb_top_k: int
) -> list[ToolDefinition]:
    knowledge_base, services = dataset.knowledge_base, dataset.services

    async def search_knowledge_base(
        query: Annotated[str, Field(min_length=2, max_length=200, description="Keywords to search runbooks for")],
        tool_context: ToolContext,
    ) -> dict:
        """Search internal operations documentation (runbooks, policies). Returns the top matching documents."""
        fault = await faults.apply(ToolName.SEARCH_KNOWLEDGE_BASE)
        q = _tokens(query)
        hits = []
        for doc in knowledge_base:
            score = (
                3 * len(q & _tokens(doc["title"])) + 2 * len(q & set(doc["tags"])) + len(q & _tokens(doc["content"]))
            )
            if score:
                hits.append(
                    {
                        "id": doc["id"],
                        "title": doc["title"],
                        "service": doc["service"],
                        "snippet": doc["content"][:300],
                        "score": float(score),
                    }
                )
        hits.sort(key=lambda h: h["score"], reverse=True)
        if fault == "malformed":
            return {"query": query, "results": [{"id": "garbage"}]}
        return {"query": query, "results": hits[:kb_top_k]}

    async def get_service_status(
        service_name: Annotated[
            str, Field(pattern=r"^[a-z0-9][a-z0-9-]{1,49}$", description="Service identifier, e.g. payment-api")
        ],
        tool_context: ToolContext,
    ) -> dict:
        """Retrieve the current health status, latency and error rate of a service."""
        fault = await faults.apply(ToolName.GET_SERVICE_STATUS)
        if service_name not in services:
            raise PermanentToolError(f"unknown service {service_name!r}; known services: {sorted(services)}")
        if fault == "malformed":
            return {"service_name": service_name, "status": "on fire"}
        return {"service_name": service_name, **services[service_name]}

    async def create_incident(
        title: Annotated[str, Field(min_length=5, max_length=120, description="Short incident title")],
        description: Annotated[
            str, Field(min_length=10, max_length=2000, description="Impact, evidence and suggested next steps")
        ],
        severity: Annotated[Severity, Field(description="Incident severity per the severity guidelines")],
        tool_context: ToolContext,
    ) -> dict:
        """Create an incident in the external incident management system. Requires human approval."""
        fault = await faults.apply(ToolName.CREATE_INCIDENT)
        # Deduplicate on the normalized title so a retried or repeated call never opens a second incident.
        dedup_key = hashlib.sha256(" ".join(sorted(_tokens(title))).encode()).hexdigest()
        incident, created = await incidents.create(title, description, severity, dedup_key)
        if fault == "malformed":
            return {"incident_id": 42}
        return {
            "incident_id": incident.id,
            "status": "created" if created else "duplicate",
            "title": incident.title,
            "severity": incident.severity,
        }

    return [
        ToolDefinition(search_knowledge_base, SearchKnowledgeBaseOutput),
        ToolDefinition(get_service_status, ServiceStatusOutput),
        ToolDefinition(create_incident, CreateIncidentOutput, requires_approval=True),
    ]
