"""MCP client adapter: mount external MCP servers as Jarvis tools.

Each server configured under ``tools.mcp_servers`` in config.yaml is
connected at startup (stdio, streamable HTTP, or SSE). Its tools are
wrapped as regular :class:`~jarvis.tools.base.Tool` objects named
``<server>_<tool>``, so the brain treats an n8n Cloud workflow exactly
like a built-in capability. ``${VAR}`` references in URLs, headers,
commands, and env values are expanded from the environment, keeping
secrets out of config.yaml.
"""

from __future__ import annotations

import os
from contextlib import AsyncExitStack
from typing import Any

import structlog

from jarvis.config import McpServerConfig
from jarvis.tools.base import Tool, ToolError

__all__ = ["McpMount", "McpRemoteTool"]


def _expand(value: str) -> str:
    return os.path.expandvars(value)


def result_to_text(result: Any) -> str:
    """Flatten an MCP call result into text for the model.

    Text blocks are joined; other block types are noted by type. An
    ``isError`` result raises :class:`ToolError` so the registry
    reports it as a failed call.
    """
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        block_type = getattr(block, "type", "unknown")
        if block_type == "text":
            parts.append(block.text)
        else:
            parts.append(f"[{block_type} content omitted]")
    text = "\n".join(parts) if parts else "(empty result)"
    if getattr(result, "isError", False):
        raise ToolError(text)
    return text


class McpRemoteTool:
    """One tool exported by a connected MCP server."""

    def __init__(
        self,
        *,
        session: Any,
        server_name: str,
        remote_name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> None:
        self.name = f"{server_name}_{remote_name}"
        self.description = description or f"Tool {remote_name} on {server_name}."
        self.input_schema = input_schema
        self._session = session
        self._remote_name = remote_name

    async def run(self, arguments: dict[str, Any]) -> str:
        """Forward the call to the remote server."""
        try:
            result = await self._session.call_tool(self._remote_name, arguments)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"MCP call {self.name} failed: {exc}") from exc
        return result_to_text(result)


class McpMount:
    """A live connection to one MCP server and its wrapped tools."""

    def __init__(self, config: McpServerConfig) -> None:
        self._config = config
        self._stack: AsyncExitStack | None = None
        self._log = structlog.get_logger("jarvis.tools.mcp")

    async def connect(self) -> list[Tool]:
        """Open the transport, initialize the session, and wrap tools."""
        try:
            from mcp import ClientSession
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ToolError(
                "The 'mcp' package is not installed; cannot mount MCP " "servers."
            ) from exc

        config = self._config
        stack = AsyncExitStack()
        try:
            if config.transport == "stdio":
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                params = StdioServerParameters(
                    command=_expand(config.command),
                    args=[_expand(a) for a in config.args],
                    env={k: _expand(v) for k, v in config.env.items()} or None,
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            elif config.transport == "sse":
                from mcp.client.sse import sse_client

                read, write = await stack.enter_async_context(
                    sse_client(
                        _expand(config.url),
                        headers={k: _expand(v) for k, v in config.headers.items()},
                    )
                )
            else:  # streamable_http
                from mcp.client.streamable_http import streamablehttp_client

                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(
                        _expand(config.url),
                        headers={k: _expand(v) for k, v in config.headers.items()},
                    )
                )

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listed = await session.list_tools()
        except ToolError:
            await stack.aclose()
            raise
        except Exception as exc:
            await stack.aclose()
            raise ToolError(
                f"Could not connect MCP server '{config.name}': {exc}"
            ) from exc

        self._stack = stack
        tools: list[Tool] = [
            McpRemoteTool(
                session=session,
                server_name=config.name,
                remote_name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in listed.tools
        ]
        self._log.info(
            "mcp_mounted",
            server=config.name,
            tools=[t.name for t in tools],
        )
        return tools

    async def aclose(self) -> None:
        """Tear the connection down."""
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
