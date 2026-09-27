"""Turns tool definitions into ADK FunctionTools plus the specs the validation plugin needs.

Adding a tool means adding a ToolDefinition where the tool lives; nothing here changes.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from google.adk.tools import FunctionTool
from pydantic import BaseModel, create_model

from harness.agent.tools.resilience import RetryPolicy, resilient


@dataclass(frozen=True)
class ToolDefinition:
    func: Callable[..., Awaitable[dict]]
    output_model: type[BaseModel]
    requires_approval: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]


def input_model_from_signature(func: Callable[..., Any]) -> type[BaseModel]:
    """Builds a Pydantic model from the tool signature so validation and the LLM declaration never drift."""
    fields: dict[str, Any] = {}
    # eval_str resolves string annotations (from `from __future__ import annotations`) in the tool's module.
    for name, param in inspect.signature(func, eval_str=True).parameters.items():
        if name == "tool_context":
            continue
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (param.annotation, default)
    return create_model(f"{func.__name__}_input", __config__={"extra": "forbid"}, **fields)


def build_tools(
    definitions: Iterable[ToolDefinition], retry: RetryPolicy
) -> tuple[list[FunctionTool], dict[str, ToolSpec]]:
    tools: list[FunctionTool] = []
    specs: dict[str, ToolSpec] = {}
    for d in definitions:
        name = d.func.__name__
        tools.append(FunctionTool(resilient(d.func, retry), require_confirmation=d.requires_approval))
        specs[name] = ToolSpec(name, input_model_from_signature(d.func), d.output_model)
    return tools, specs
