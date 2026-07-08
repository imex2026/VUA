"""File operations tool, sandboxed to one root directory."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from jarvis.tools.base import ToolError

__all__ = ["FileOpsTool"]

_MAX_READ_CHARS = 50_000


class FileOpsTool:
    """List, read, and write text files inside a sandbox folder.

    Paths are resolved and verified to stay under the sandbox root, so
    the model cannot touch anything else on disk.
    """

    name = "files"
    description = (
        "Work with text files in Jarvis's sandbox folder: 'list' a "
        "directory, 'read' a file, or 'write' content to a file "
        "(creating parent folders as needed). Paths are relative to "
        "the sandbox root."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "read", "write"],
                "description": "What to do.",
            },
            "path": {
                "type": "string",
                "description": "Relative path ('' or '.' lists the root).",
            },
            "content": {
                "type": "string",
                "description": "Text to write (write).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative: str) -> Path:
        candidate = (self._root / relative).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise ToolError(
                f"Path {relative!r} escapes the sandbox; use relative "
                "paths inside it."
            )
        return candidate

    async def run(self, arguments: dict[str, Any]) -> str:
        """Dispatch to the requested action in a worker thread."""
        action = arguments.get("action")
        relative = str(arguments.get("path", "") or ".")
        if action == "list":
            return await asyncio.to_thread(self._list, relative)
        if action == "read":
            return await asyncio.to_thread(self._read, relative)
        if action == "write":
            content = arguments.get("content")
            if not isinstance(content, str):
                raise ToolError("'write' requires string 'content'.")
            return await asyncio.to_thread(self._write, relative, content)
        raise ToolError(f"Unknown files action: {action!r}")

    def _list(self, relative: str) -> str:
        target = self._resolve(relative)
        if not target.is_dir():
            raise ToolError(f"{relative!r} is not a directory.")
        entries = sorted(target.iterdir(), key=lambda p: p.name)
        if not entries:
            return "(empty directory)"
        lines = []
        for entry in entries:
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                lines.append(f"{entry.name} ({entry.stat().st_size} bytes)")
        return "\n".join(lines)

    def _read(self, relative: str) -> str:
        target = self._resolve(relative)
        if not target.is_file():
            raise ToolError(f"File not found: {relative!r}.")
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ToolError(f"Could not read {relative!r}: {exc}") from exc
        if len(text) > _MAX_READ_CHARS:
            return (
                text[:_MAX_READ_CHARS] + f"\n... (truncated, {len(text)} chars total)"
            )
        return text

    def _write(self, relative: str, content: str) -> str:
        target = self._resolve(relative)
        if target.is_dir():
            raise ToolError(f"{relative!r} is a directory.")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Could not write {relative!r}: {exc}") from exc
        return f"Wrote {len(content)} chars to {relative!r}."
