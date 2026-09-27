"""RuleBasedLlm: a deterministic planner implementing ADK's BaseLlm, for demos without cloud access."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from harness.adapters.llm.responses import call, text
from harness.constants import ToolName


def _tool_results(llm_request: LlmRequest) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for content in llm_request.contents:
        for part in content.parts or []:
            if part.function_response and part.function_response.name:
                results[part.function_response.name] = part.function_response.response or {}
    return results


def _objective(llm_request: LlmRequest) -> str:
    for content in llm_request.contents:
        if content.role == "user":
            for part in content.parts or []:
                if part.text:
                    return part.text
    return ""


class RuleBasedLlm(BaseLlm):
    """Search runbooks -> check the service -> propose an incident if it is unhealthy -> summarize."""

    model: str = "rule-based"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield self._decide(llm_request)

    def _decide(self, req: LlmRequest) -> LlmResponse:
        objective = _objective(req)
        results = _tool_results(req)

        kb = results.get(ToolName.SEARCH_KNOWLEDGE_BASE)
        if kb is None:
            return call(ToolName.SEARCH_KNOWLEDGE_BASE, query=objective[:200])

        service = _find_service(objective, kb)
        status = results.get(ToolName.GET_SERVICE_STATUS)
        if service and status is None:
            return call(ToolName.GET_SERVICE_STATUS, service_name=service)

        incident = results.get(ToolName.CREATE_INCIDENT)
        unhealthy = status and status.get("status") in ("degraded", "down")
        if unhealthy and incident is None:
            severity = (
                "critical"
                if status["status"] == "down"
                else ("high" if status.get("error_rate_pct", 0) >= 2 else "medium")
            )
            runbook = (kb.get("results") or [{}])[0].get("title", "n/a")
            return call(
                ToolName.CREATE_INCIDENT,
                title=f"{service} {status['status']}: {status.get('message', '')}"[:120],
                description=(
                    f"Status={status['status']}, p95={status.get('latency_p95_ms')}ms, "
                    f"errors={status.get('error_rate_pct')}%. Runbook: {runbook}. Reported objective: {objective[:300]}"
                ),
                severity=severity,
            )
        return text(_summary(objective, service, kb, status, incident))


def _find_service(objective: str, kb: dict) -> str | None:
    match = re.search(r"\b([a-z]+(?:-[a-z]+)+)\b", objective.lower())
    if match:
        return match.group(1)
    for hit in kb.get("results") or []:
        if hit.get("service"):
            return hit["service"]
    return None


def _summary(objective: str, service: str | None, kb: dict, status: dict | None, incident: dict | None) -> str:
    lines = [f"Objective: {objective}"]
    hits = kb.get("results") or []
    lines.append(
        "Runbooks: " + (", ".join(f"{h['id']} {h['title']}" for h in hits) if hits else kb.get("error", "none found"))
    )
    if status is not None:
        if "error" in status:
            lines.append(f"Status of {service}: unavailable ({status['error']})")
        else:
            lines.append(f"Status of {service}: {status['status']} - {status.get('message')}")
    if incident is not None:
        if "error" in incident:
            reason = f" Reason: {incident['reason']}" if incident.get("reason") else ""
            lines.append(f"Incident not created: {incident['error']}{reason}")
        else:
            lines.append(f"Incident {incident['incident_id']} ({incident['status']}, severity {incident['severity']}).")
    elif status and status.get("status") == "healthy":
        lines.append("Service is healthy; no incident needed.")
    return "\n".join(lines)
