"""Tells the model who rejected a tool call and why.

ADK answers a rejected call with a fixed "This tool call is rejected." message. The harness
sends the operator's decision in the ToolConfirmation payload; this plugin surfaces it so the
agent can explain the outcome instead of guessing (or trying again).
"""

from __future__ import annotations

from typing import Any

from google.adk.plugins import BasePlugin
from google.adk.tools import BaseTool, ToolContext

from harness.constants import ToolErrorType


class ApprovalFeedbackPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="harness_approval_feedback")

    async def after_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext, result: dict
    ) -> dict | None:
        confirmation = tool_context.tool_confirmation
        if confirmation is None or confirmation.confirmed:
            return None
        payload = confirmation.payload if isinstance(confirmation.payload, dict) else {}
        return {
            "error": f"The operator rejected this {tool.name} call. Do not retry it.",
            "error_type": ToolErrorType.REJECTED_BY_OPERATOR,
            "rejected_by": payload.get("decided_by"),
            "reason": payload.get("reason") or "no reason given",
        }
