"""Execution limits ADK does not provide out of the box.

ADK already enforces RunConfig.max_llm_calls. This plugin adds:
  * a wall-clock budget per invocation (human approval wait time is not counted,
    because resuming after approval starts a new invocation),
  * a total tool-call budget per run,
  * blocking of repeated identical tool calls (the typical "stuck agent" loop).

When a hard limit trips, the reason is written to session state and the next model
call is short-circuited with a final message, so the run ends cleanly instead of
being killed mid-step.
"""

from __future__ import annotations

import json
import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

from harness.constants import StateKey, TerminationReason, ToolErrorType, TraceEventType
from harness.observability import Tracer


class GuardrailPlugin(BasePlugin):
    def __init__(
        self,
        tracer: Tracer,
        *,
        time_budget_s: float,
        max_tool_calls: int,
        max_identical_tool_calls: int,
    ):
        super().__init__(name="harness_guardrails")
        self.tracer = tracer
        self.time_budget_s = time_budget_s
        self.max_tool_calls = max_tool_calls
        self.max_identical_tool_calls = max_identical_tool_calls
        self._invocation_started: dict[str, float] = {}

    async def before_run_callback(self, *, invocation_context: InvocationContext) -> None:
        self._invocation_started[invocation_context.invocation_id] = time.monotonic()
        return None

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        self._invocation_started.pop(invocation_context.invocation_id, None)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        reason = callback_context.state.get(StateKey.TERMINATION)
        if reason is None:
            started = self._invocation_started.get(callback_context.invocation_id)
            if started is not None and time.monotonic() - started > self.time_budget_s:
                reason = TerminationReason.TIME_BUDGET
                await self._trip(callback_context, reason, budget_s=self.time_budget_s)
        if reason is None:
            return None
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"[harness] Run stopped before completion: {reason}.")],
            )
        )

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict | None:
        # A call resumed after human approval was already counted when it was first proposed.
        if tool_context.tool_confirmation is not None:
            return None

        total = tool_context.state.get(StateKey.TOOL_CALLS, 0) + 1
        tool_context.state[StateKey.TOOL_CALLS] = total
        if total > self.max_tool_calls:
            await self._trip(tool_context, TerminationReason.MAX_TOOL_CALLS, limit=self.max_tool_calls)
            return {
                "error": "Tool call budget exhausted; the run is being stopped.",
                "error_type": ToolErrorType.LIMIT_EXCEEDED,
            }

        key = f"{tool.name}:{json.dumps(tool_args, sort_keys=True, default=str)}"
        counts = dict(tool_context.state.get(StateKey.CALL_COUNTS, {}))
        counts[key] = counts.get(key, 0) + 1
        tool_context.state[StateKey.CALL_COUNTS] = counts
        if counts[key] > self.max_identical_tool_calls:
            await self.tracer.emit(
                tool_context.session.id,
                TraceEventType.LOOP_DETECTED,
                name=tool.name,
                invocation_id=tool_context.invocation_id,
                args=tool_args,
                count=counts[key],
            )
            return {
                "error": (
                    f"Blocked: {tool.name} was already called {counts[key] - 1} times with identical arguments. "
                    "Use the earlier result or change approach."
                ),
                "error_type": ToolErrorType.REPEATED_CALL,
            }
        return None

    async def _trip(self, ctx: CallbackContext, reason: str, **details: Any) -> None:
        ctx.state[StateKey.TERMINATION] = reason
        await self.tracer.emit(
            ctx.session.id, TraceEventType.LIMIT_EXCEEDED, name=reason, invocation_id=ctx.invocation_id, **details
        )
