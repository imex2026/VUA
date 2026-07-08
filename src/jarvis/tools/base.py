"""The Tool protocol and registry.

Every capability Jarvis can invoke - built-in or mounted from an MCP
server - implements the same small protocol: a name, a description, a
JSON schema, and an async ``run``. The registry turns those into
Anthropic tool specs and executes calls, converting failures into
error results the model can react to instead of crashes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import structlog

__all__ = ["Tool", "ToolError", "ToolRegistry", "ToolResult"]

# Anthropic's tool-name constraint.
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ToolError(RuntimeError):
    """A tool failed in an expected way; the message goes to the model."""


@runtime_checkable
class Tool(Protocol):
    """A self-contained capability the model can call."""

    name: str
    description: str
    input_schema: dict[str, Any]

    async def run(self, arguments: dict[str, Any]) -> str:
        """Execute with schema-shaped ``arguments``; return text for
        the model. Raise :class:`ToolError` for expected failures."""
        ...


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool call produced, error or not."""

    content: str
    is_error: bool = False


class ToolRegistry:
    """Holds tools, exposes their specs, and executes calls safely."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._log = structlog.get_logger("jarvis.tools")

    def register(self, tool: Tool) -> None:
        """Add a tool; names must be unique and API-safe."""
        if not _NAME_RE.match(tool.name):
            raise ValueError(
                f"Invalid tool name {tool.name!r}: must match {_NAME_RE.pattern}"
            )
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        self._log.info("tool_registered", tool=tool.name)

    def get(self, name: str) -> Tool | None:
        """Return the tool called ``name``, if registered."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Registered tool names, in registration order."""
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        """Anthropic-format tool specifications."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a tool call, never raising: failures become error results
        so the model can recover or apologize."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"Unknown tool: {name}", is_error=True)
        try:
            content = await tool.run(arguments)
            return ToolResult(content)
        except ToolError as exc:
            self._log.warning("tool_error", tool=name, error=str(exc))
            return ToolResult(str(exc), is_error=True)
        except Exception as exc:
            self._log.exception("tool_crashed", tool=name)
            return ToolResult(f"Tool '{name}' crashed: {exc}", is_error=True)
