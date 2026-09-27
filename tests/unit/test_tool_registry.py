from typing import Annotated, Literal

import pytest
from google.adk.tools import ToolContext
from pydantic import BaseModel, Field, ValidationError

from harness.agent.tools.registry import ToolDefinition, build_tools, input_model_from_signature
from harness.agent.tools.resilience import RetryPolicy


class Out(BaseModel):
    ok: bool


async def open_ticket(
    title: Annotated[str, Field(min_length=5)],
    severity: Literal["low", "high"],
    tool_context: ToolContext,
    note: str = "",
) -> dict:
    """Open a ticket."""
    return {"ok": True}


def test_input_model_mirrors_the_signature():
    model = input_model_from_signature(open_ticket)

    assert set(model.model_fields) == {"title", "severity", "note"}  # tool_context is injected by ADK, not the LLM
    assert model.model_validate({"title": "disk full", "severity": "high"}).note == ""


@pytest.mark.parametrize(
    "args",
    [
        {"title": "x", "severity": "high"},  # too short
        {"title": "disk full", "severity": "urgent"},  # not in the enum
        {"title": "disk full", "severity": "high", "region": "eu"},  # unexpected argument
        {"severity": "high"},  # missing required
    ],
)
def test_input_model_rejects_bad_arguments(args):
    with pytest.raises(ValidationError):
        input_model_from_signature(open_ticket).model_validate(args)


def test_build_tools_applies_approval_flag_and_specs():
    tools, specs = build_tools([ToolDefinition(open_ticket, Out, requires_approval=True)], RetryPolicy(1, 1, 0))

    assert [t.name for t in tools] == ["open_ticket"]
    assert tools[0]._require_confirmation is True
    assert specs["open_ticket"].output_model is Out
