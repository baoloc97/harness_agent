"""Shared fixtures. Tests use ScriptedLlm (no network) and a throwaway SQLite database.

Set HARNESS_TEST_DATABASE_URL to run the same suite against Postgres.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from harness.adapters.llm import ScriptedLlm, Step
from harness.agent.tools.resilience import FaultInjector
from harness.bootstrap import Harness, build_harness
from harness.config import Settings
from harness.service_layer.service import HarnessService
from harness.views import RunView, TraceEventView

FAST_SETTINGS = dict(
    llm_backend="rule_based",
    tool_timeout_s=0.2,
    tool_max_attempts=3,
    tool_backoff_base_s=0.01,
    run_time_budget_s=10,
    max_llm_calls=12,
    max_tool_calls=10,
    max_identical_tool_calls=2,
    max_model_retries=2,
)


@pytest.fixture
async def database_url(tmp_path) -> str:
    url = os.environ.get("HARNESS_TEST_DATABASE_URL")
    if not url:
        return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    # Shared Postgres test database: start every test from an empty schema.
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await engine.dispose()
    return url


def settings_for(database_url: str, **overrides: Any) -> Settings:
    return Settings(_env_file=None, database_url=database_url, **{**FAST_SETTINGS, **overrides})


@pytest.fixture
async def make_service(database_url):
    """Builds a started harness around a ScriptedLlm (or a given llm) and returns its service."""
    started: list[Harness] = []

    async def factory(
        steps: list[Step] | None = None, *, llm: Any = None, faults: FaultInjector | None = None, **overrides: Any
    ) -> HarnessService:
        model = llm if llm is not None else (ScriptedLlm(steps) if steps is not None else None)
        harness = build_harness(settings_for(database_url, **overrides), llm=model, faults=faults or FaultInjector())
        await harness.start()
        started.append(harness)
        return harness.service

    yield factory
    for harness in started:
        await harness.stop()


def tool_results(run: RunView, tool: str) -> list[dict]:
    return [h.result for h in run.history if h.type == "tool_result" and h.tool == tool]


def trace_of(trace: list[TraceEventView], type_: str, name: str | None = None) -> list[TraceEventView]:
    return [t for t in trace if t.type == type_ and (name is None or t.name == name)]


def unique_title() -> str:
    return f"payment-api degraded {uuid.uuid4().hex[:6]}"
