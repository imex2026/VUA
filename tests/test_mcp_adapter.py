"""Unit tests for the MCP adapter's tool wrapping (no real servers)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.tools.base import ToolError, ToolRegistry
from jarvis.tools.mcp_adapter import McpRemoteTool, result_to_text


class FakeSession:
    def __init__(self, result: Any = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        if self.error is not None:
            raise self.error
        return self.result


def _text_result(*texts: str, is_error: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=t) for t in texts],
        isError=is_error,
    )


def _tool(session: FakeSession) -> McpRemoteTool:
    return McpRemoteTool(
        session=session,
        server_name="n8n",
        remote_name="run_workflow",
        description="Runs a workflow.",
        input_schema={"type": "object", "properties": {}},
    )


def test_remote_tool_name_is_prefixed() -> None:
    tool = _tool(FakeSession())

    assert tool.name == "n8n_run_workflow"
    assert tool.description == "Runs a workflow."


async def test_run_forwards_and_extracts_text() -> None:
    session = FakeSession(result=_text_result("done", "extra line"))
    tool = _tool(session)

    output = await tool.run({"workflow": "daily"})

    assert output == "done\nextra line"
    # The *remote* (unprefixed) name is what goes over the wire.
    assert session.calls == [("run_workflow", {"workflow": "daily"})]


async def test_error_result_raises_tool_error() -> None:
    session = FakeSession(result=_text_result("workflow crashed", is_error=True))

    with pytest.raises(ToolError, match="workflow crashed"):
        await _tool(session).run({})


async def test_transport_failure_raises_tool_error() -> None:
    session = FakeSession(error=ConnectionError("socket closed"))

    with pytest.raises(ToolError, match="socket closed"):
        await _tool(session).run({})


async def test_registry_integration() -> None:
    registry = ToolRegistry()
    registry.register(_tool(FakeSession(result=_text_result("ok"))))

    result = await registry.execute("n8n_run_workflow", {})

    assert not result.is_error
    assert result.content == "ok"


def test_result_to_text_handles_non_text_blocks() -> None:
    result = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="caption"),
            SimpleNamespace(type="image", data="..."),
        ],
        isError=False,
    )

    assert result_to_text(result) == "caption\n[image content omitted]"


def test_result_to_text_empty() -> None:
    assert result_to_text(SimpleNamespace(content=[], isError=False)) == (
        "(empty result)"
    )
