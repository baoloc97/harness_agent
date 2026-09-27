"""Records the lifecycle of every model and tool call as trace events.

ADK also emits OpenTelemetry spans; this plugin adds a queryable per-run trace in
Postgres (served by GET /runs/{id}/trace) with latency, retries and token usage.
"""

from __future__ import annotations

import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from google.adk.tools import BaseTool, ToolContext

from harness.agent.tools.resilience import attempts_key
from harness.constants import TraceEventType
from harness.observability import Tracer


class TracingPlugin(BasePlugin):
    def __init__(self, tracer: Tracer):
        super().__init__(name="harness_tracing")
        self.tracer = tracer
        self._model_started: dict[str, float] = {}
        self._tool_started: dict[str, float] = {}

    async def before_run_callback(self, *, invocation_context: InvocationContext) -> None:
        await self.tracer.emit(
            invocation_context.session.id,
            TraceEventType.INVOCATION_START,
            invocation_id=invocation_context.invocation_id,
        )
        return None

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        await self.tracer.emit(
            invocation_context.session.id, TraceEventType.INVOCATION_END, invocation_id=invocation_context.invocation_id
        )

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        self._model_started[callback_context.invocation_id] = time.perf_counter()
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        started = self._model_started.pop(callback_context.invocation_id, None)
        parts = llm_response.content.parts if llm_response.content and llm_response.content.parts else []
        usage = llm_response.usage_metadata
        await self.tracer.emit(
            callback_context.session.id,
            TraceEventType.MODEL_CALL,
            invocation_id=callback_context.invocation_id,
            latency_ms=_ms_since(started),
            function_calls=[
                {"name": p.function_call.name, "args": p.function_call.args} for p in parts if p.function_call
            ],
            text=" ".join(p.text for p in parts if p.text) or None,
            finish_reason=str(llm_response.finish_reason) if llm_response.finish_reason else None,
            error_code=llm_response.error_code,
            error_message=llm_response.error_message,
            prompt_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
        )
        return None

    async def on_model_error_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest, error: Exception
    ) -> LlmResponse | None:
        await self.tracer.emit(
            callback_context.session.id,
            TraceEventType.MODEL_ERROR,
            invocation_id=callback_context.invocation_id,
            error=f"{type(error).__name__}: {error}",
        )
        return None

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict | None:
        self._tool_started[tool_context.function_call_id or tool.name] = time.perf_counter()
        return None

    async def after_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext, result: dict
    ) -> dict | None:
        started = self._tool_started.pop(tool_context.function_call_id or tool.name, None)
        is_error = isinstance(result, dict) and ("error" in result or "response_type" in result)
        # Set by the resilience wrapper only when the tool body actually ran; absent for calls
        # answered without executing (validation/guardrail blocks, the approval placeholder).
        attempts = tool_context.state.get(attempts_key(tool_context.function_call_id))
        await self.tracer.emit(
            tool_context.session.id,
            TraceEventType.TOOL_CALL,
            name=tool.name,
            invocation_id=tool_context.invocation_id,
            latency_ms=_ms_since(started),
            status="error" if is_error else "ok",
            args=tool_args,
            result=result,
            attempts=attempts,
            executed=attempts is not None,
            approved=tool_context.tool_confirmation.confirmed if tool_context.tool_confirmation else None,
        )
        return None

    async def on_tool_error_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext, error: Exception
    ) -> dict | None:
        await self.tracer.emit(
            tool_context.session.id,
            TraceEventType.TOOL_ERROR,
            name=tool.name,
            invocation_id=tool_context.invocation_id,
            args=tool_args,
            error=f"{type(error).__name__}: {error}",
            attempts=tool_context.state.get(attempts_key(tool_context.function_call_id)),
        )
        return None


def _ms_since(started: float | None) -> float | None:
    return round((time.perf_counter() - started) * 1000, 1) if started is not None else None
