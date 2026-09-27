"""Composition root: the only place that knows how the concrete pieces fit together.

The API, the CLI and the tests all build the harness through build_harness().
"""

from __future__ import annotations

from dataclasses import dataclass

from google.adk.models.base_llm import BaseLlm
from google.adk.plugins import ReflectAndRetryModelPlugin, ReflectAndRetryToolPlugin
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService

from harness.adapters.incidents import MockIncidentGateway
from harness.adapters.llm import build_llm
from harness.adapters.orm import Database
from harness.adapters.repository import ApprovalRepository, RunRepository, TraceRepository
from harness.agent.agent import build_app
from harness.agent.plugins.approval import ApprovalFeedbackPlugin
from harness.agent.plugins.guardrails import GuardrailPlugin
from harness.agent.plugins.tracing import TracingPlugin
from harness.agent.plugins.validation import ToolValidationPlugin
from harness.agent.tools.mock_tools import MockDataset, mock_tool_definitions
from harness.agent.tools.registry import build_tools
from harness.agent.tools.resilience import FaultInjector, RetryPolicy
from harness.config import Settings
from harness.observability import Tracer
from harness.service_layer.orchestrator import RunLimits, RunOrchestrator
from harness.service_layer.service import HarnessService


@dataclass
class Harness:
    """A wired harness plus the resources whose lifecycle it owns."""

    service: HarnessService
    faults: FaultInjector  # demo/test hook for the mock tools; deliberately not part of the service
    database: Database
    runner: Runner

    async def start(self) -> None:
        await self.database.create_schema()

    async def stop(self) -> None:
        await self.runner.close()
        await self.database.close()


def build_harness(settings: Settings, *, llm: BaseLlm | None = None, faults: FaultInjector | None = None) -> Harness:
    database = Database(settings.database_url)
    runs = RunRepository(database.sessions)
    approvals = ApprovalRepository(database.sessions)
    traces = TraceRepository(database.sessions)
    incidents = MockIncidentGateway(database.sessions)
    tracer = Tracer(traces, max_str_len=settings.trace_max_str_len)
    faults = faults or FaultInjector.from_string(settings.mock_faults)

    retry = RetryPolicy(settings.tool_timeout_s, settings.tool_max_attempts, settings.tool_backoff_base_s)
    tools, specs = build_tools(
        mock_tool_definitions(MockDataset.load(settings.data_dir), incidents, faults, kb_top_k=settings.kb_top_k), retry
    )

    # Order matters: tracing observes first; validation runs before guardrails so rejected
    # calls do not consume budget; ADK's reflect-and-retry plugins go last.
    plugins = [
        TracingPlugin(tracer),
        ApprovalFeedbackPlugin(),
        ToolValidationPlugin(specs, tracer),
        GuardrailPlugin(
            tracer,
            time_budget_s=settings.run_time_budget_s,
            max_tool_calls=settings.max_tool_calls,
            max_identical_tool_calls=settings.max_identical_tool_calls,
        ),
        ReflectAndRetryToolPlugin(
            max_retries=settings.tool_reflection_retries, throw_exception_if_retry_exceeded=False
        ),
        ReflectAndRetryModelPlugin(max_retries=settings.max_model_retries, throw_exception_if_retry_exceeded=False),
    ]
    app = build_app(llm or build_llm(settings), tools, plugins, temperature=settings.model_temperature)
    sessions = DatabaseSessionService(db_url=settings.database_url)
    runner = Runner(app=app, session_service=sessions)

    limits = RunLimits(
        max_llm_calls=settings.max_llm_calls,
        hard_timeout_s=settings.run_time_budget_s + settings.tool_timeout_s * settings.tool_max_attempts,
    )
    orchestrator = RunOrchestrator(runner, sessions, runs, approvals, traces, tracer, limits)
    service = HarnessService(orchestrator, sessions, runs, approvals, traces, incidents, tracer)
    return Harness(service=service, faults=faults, database=database, runner=runner)
