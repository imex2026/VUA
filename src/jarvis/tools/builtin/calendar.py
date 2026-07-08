"""Local calendar / reminders tool backed by a JSON file."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from jarvis.tools.base import ToolError

__all__ = ["CalendarTool"]


class CalendarTool:
    """Add, list, and remove calendar entries / reminders.

    Entries live in a JSON file so they survive restarts; file access
    runs in a worker thread guarded by a lock.
    """

    name = "calendar"
    description = (
        "Manage the user's local calendar and reminders. Actions: "
        "'add' (title + when, ISO 8601 like 2026-07-09T18:30), "
        "'list' (all upcoming entries), 'remove' (by id)."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "remove"],
                "description": "What to do.",
            },
            "title": {"type": "string", "description": "Event title (add)."},
            "when": {
                "type": "string",
                "description": "ISO 8601 date/time (add).",
            },
            "notes": {"type": "string", "description": "Optional notes (add)."},
            "id": {"type": "string", "description": "Entry id (remove)."},
        },
        "required": ["action"],
    }

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def run(self, arguments: dict[str, Any]) -> str:
        """Dispatch to the requested action."""
        action = arguments.get("action")
        if action == "add":
            return await self._add(arguments)
        if action == "list":
            return await self._list()
        if action == "remove":
            return await self._remove(arguments)
        raise ToolError(f"Unknown calendar action: {action!r}")

    def _load(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolError(f"Could not read calendar file: {exc}") from exc
        return data if isinstance(data, list) else []

    def _save(self, entries: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    async def _add(self, arguments: dict[str, Any]) -> str:
        title = str(arguments.get("title", "")).strip()
        when_raw = str(arguments.get("when", "")).strip()
        if not title:
            raise ToolError("Adding an entry requires a 'title'.")
        try:
            when = datetime.fromisoformat(when_raw)
        except ValueError as exc:
            raise ToolError(
                f"'when' must be ISO 8601 (got {when_raw!r}), " "e.g. 2026-07-09T18:30."
            ) from exc
        entry = {
            "id": uuid.uuid4().hex[:8],
            "title": title,
            "when": when.isoformat(),
            "notes": str(arguments.get("notes", "")),
        }

        async with self._lock:

            def _write() -> None:
                entries = self._load()
                entries.append(entry)
                self._save(entries)

            await asyncio.to_thread(_write)
        return f"Added '{title}' at {entry['when']} (id {entry['id']})."

    async def _list(self) -> str:
        async with self._lock:
            entries = await asyncio.to_thread(self._load)
        if not entries:
            return "The calendar is empty."
        entries.sort(key=lambda e: str(e.get("when", "")))
        now = datetime.now()
        lines = []
        for entry in entries:
            try:
                past = datetime.fromisoformat(str(entry["when"])) < now
            except ValueError:
                past = False
            marker = " (past)" if past else ""
            notes = f" - {entry['notes']}" if entry.get("notes") else ""
            lines.append(
                f"[{entry['id']}] {entry['when']}{marker}: " f"{entry['title']}{notes}"
            )
        return "\n".join(lines)

    async def _remove(self, arguments: dict[str, Any]) -> str:
        entry_id = str(arguments.get("id", "")).strip()
        if not entry_id:
            raise ToolError("Removing an entry requires its 'id'.")

        async with self._lock:

            def _remove_entry() -> bool:
                entries = self._load()
                remaining = [e for e in entries if e.get("id") != entry_id]
                if len(remaining) == len(entries):
                    return False
                self._save(remaining)
                return True

            removed = await asyncio.to_thread(_remove_entry)
        if not removed:
            raise ToolError(f"No calendar entry with id {entry_id!r}.")
        return f"Removed entry {entry_id}."
