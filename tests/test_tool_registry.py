"""Unit tests for the tool registry."""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.tools.base import ToolError, ToolRegistry


class EchoTool:
    """Echoes its 'text' argument; the standard test tool."""

    name = "echo"
    description = "Echo the given text."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run(self, arguments: dict[str, Any]) -> str:
        self.calls.append(arguments)
        return f"echo: {arguments.get('text', '')}"


class FailingTool:
    name = "failing"
    description = "Always fails."
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, exception: Exception) -> None:
        self._exception = exception

    async def run(self, arguments: dict[str, Any]) -> str:
        raise self._exception


def test_register_and_specs() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    assert registry.names() == ["echo"]
    assert len(registry) == 1
    spec = registry.specs()[0]
    assert spec == {
        "name": "echo",
        "description": "Echo the given text.",
        "input_schema": EchoTool.input_schema,
    }


def test_duplicate_name_rejected() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(EchoTool())


def test_invalid_name_rejected() -> None:
    tool = EchoTool()
    tool.name = "not a valid name!"
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Invalid tool name"):
        registry.register(tool)


async def test_execute_success() -> None:
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)

    result = await registry.execute("echo", {"text": "hello"})

    assert not result.is_error
    assert result.content == "echo: hello"
    assert tool.calls == [{"text": "hello"}]


async def test_execute_unknown_tool_is_error_result() -> None:
    registry = ToolRegistry()

    result = await registry.execute("nope", {})

    assert result.is_error
    assert "Unknown tool" in result.content


async def test_tool_error_becomes_error_result() -> None:
    registry = ToolRegistry()
    registry.register(FailingTool(ToolError("bad input")))

    result = await registry.execute("failing", {})

    assert result.is_error
    assert result.content == "bad input"


async def test_crash_becomes_error_result() -> None:
    registry = ToolRegistry()
    registry.register(FailingTool(RuntimeError("kaboom")))

    result = await registry.execute("failing", {})

    assert result.is_error
    assert "crashed" in result.content and "kaboom" in result.content
