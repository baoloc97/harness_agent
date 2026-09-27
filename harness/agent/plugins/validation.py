"""Validates tool inputs before the approval gate and tool outputs before the model sees them.

Why not rely on ADK alone: ADK's argument validation (FUNCTION_TOOL_ARG_VALIDATION) runs
inside FunctionTool.run_async, i.e. *after* the confirmation gate, so a human could be asked
to approve a call that is going to be rejected anyway. ADK does not validate outputs at all.
"""

from __future__ import annotations

from typing import Any

from google.adk.plugins import BasePlugin
from google.adk.tools import BaseTool, ToolContext
from pydantic import ValidationError

from harness.agent.tools.registry import ToolSpec
from harness.constants import ToolErrorType, TraceEventType
from harness.observability import Tracer


def _format_errors(e: ValidationError) -> list[str]:
    return [f"{'.'.join(map(str, err['loc'])) or '<root>'}: {err['msg']}" for err in e.errors()]


class ToolValidationPlugin(BasePlugin):
    def __init__(self, specs: dict[str, ToolSpec], tracer: Tracer):
        super().__init__(name="harness_tool_validation")
        self.specs = specs
        self.tracer = tracer

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict | None:
        spec = self.specs.get(tool.name)
        if spec is None:
            return None  # unknown tools are answered by ADK's tool-not-found handling
        try:
            validated = spec.input_model.model_validate(tool_args)
        except ValidationError as e:
            errors = _format_errors(e)
            await self.tracer.emit(
                tool_context.session.id,
                TraceEventType.TOOL_INPUT_INVALID,
                name=tool.name,
                invocation_id=tool_context.invocation_id,
                args=tool_args,
                errors=errors,
            )
            return {
                "error": f"Invalid arguments for {tool.name}; the tool was not executed.",
                "error_type": ToolErrorType.INVALID_ARGUMENTS,
                "details": errors,
            }
        # Hand the tool the coerced values (e.g. stripped/normalized types).
        tool_args.clear()
        tool_args.update(validated.model_dump())
        return None

    async def after_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext, result: dict
    ) -> dict | None:
        spec = self.specs.get(tool.name)
        if spec is None or not isinstance(result, dict) or "error" in result or "response_type" in result:
            return None  # errors/guidance produced by the harness or ADK, not tool output
        try:
            spec.output_model.model_validate(result)
        except ValidationError as e:
            errors = _format_errors(e)
            await self.tracer.emit(
                tool_context.session.id,
                TraceEventType.TOOL_OUTPUT_INVALID,
                name=tool.name,
                invocation_id=tool_context.invocation_id,
                output=result,
                errors=errors,
            )
            return {
                "error": f"{tool.name} returned a malformed response; its data cannot be trusted.",
                "error_type": ToolErrorType.INVALID_TOOL_OUTPUT,
            }
        return None
