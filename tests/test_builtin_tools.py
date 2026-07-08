"""Unit tests for the built-in tools (no network, tmp-dir sandboxes)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.tools.base import ToolError
from jarvis.tools.builtin.calendar import CalendarTool
from jarvis.tools.builtin.files import FileOpsTool
from jarvis.tools.builtin.smart_home import SmartHomeTool
from jarvis.tools.builtin.web_search import WebSearchTool

# --- calendar ---------------------------------------------------------


async def test_calendar_add_list_remove(tmp_path: Path) -> None:
    tool = CalendarTool(tmp_path / "cal.json")

    added = await tool.run(
        {"action": "add", "title": "Dentist", "when": "2026-08-01T09:00"}
    )
    assert "Dentist" in added
    entry_id = added.split("id ")[1].rstrip(").")

    listing = await tool.run({"action": "list"})
    assert "Dentist" in listing and entry_id in listing

    removed = await tool.run({"action": "remove", "id": entry_id})
    assert entry_id in removed
    assert "empty" in await tool.run({"action": "list"})


async def test_calendar_rejects_bad_datetime(tmp_path: Path) -> None:
    tool = CalendarTool(tmp_path / "cal.json")

    with pytest.raises(ToolError, match="ISO 8601"):
        await tool.run({"action": "add", "title": "X", "when": "next tuesday"})


async def test_calendar_remove_missing_id(tmp_path: Path) -> None:
    tool = CalendarTool(tmp_path / "cal.json")

    with pytest.raises(ToolError, match="No calendar entry"):
        await tool.run({"action": "remove", "id": "nope1234"})


async def test_calendar_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "cal.json"
    await CalendarTool(path).run(
        {"action": "add", "title": "Standup", "when": "2026-08-02T10:00"}
    )

    listing = await CalendarTool(path).run({"action": "list"})
    assert "Standup" in listing


# --- files ------------------------------------------------------------


async def test_files_write_read_list(tmp_path: Path) -> None:
    tool = FileOpsTool(tmp_path / "sandbox")

    await tool.run({"action": "write", "path": "notes/todo.txt", "content": "buy milk"})
    assert await tool.run({"action": "read", "path": "notes/todo.txt"}) == ("buy milk")
    listing = await tool.run({"action": "list", "path": "notes"})
    assert "todo.txt" in listing


async def test_files_sandbox_escape_rejected(tmp_path: Path) -> None:
    tool = FileOpsTool(tmp_path / "sandbox")

    with pytest.raises(ToolError, match="escapes the sandbox"):
        await tool.run({"action": "read", "path": "../secrets.txt"})


async def test_files_read_missing(tmp_path: Path) -> None:
    tool = FileOpsTool(tmp_path / "sandbox")

    with pytest.raises(ToolError, match="not found"):
        await tool.run({"action": "read", "path": "ghost.txt"})


# --- smart home -------------------------------------------------------


async def test_smart_home_list_and_set() -> None:
    tool = SmartHomeTool()

    listing = await tool.run({"action": "list"})
    assert "living_room_light: off" in listing

    result = await tool.run(
        {"action": "set", "device": "living_room_light", "state": "on"}
    )
    assert "simulated" in result
    assert "living_room_light: on" in await tool.run({"action": "list"})


async def test_smart_home_unknown_device() -> None:
    tool = SmartHomeTool()

    with pytest.raises(ToolError, match="Unknown device"):
        await tool.run({"action": "set", "device": "jacuzzi", "state": "on"})


# --- web search formatting (network paths are not tested) --------------


def test_format_brave_results() -> None:
    data = {
        "web": {
            "results": [
                {
                    "title": "Python",
                    "url": "https://python.org",
                    "description": "The language.",
                },
                {"title": "Docs", "url": "https://docs.python.org"},
            ]
        }
    }

    text = WebSearchTool.format_brave(data, count=5)

    assert "1. Python" in text and "https://python.org" in text
    assert "2. Docs" in text


def test_format_brave_empty() -> None:
    assert WebSearchTool.format_brave({}, count=5) == "No results found."


def test_format_duckduckgo_results() -> None:
    data = {
        "AbstractText": "Python is a programming language.",
        "AbstractURL": "https://en.wikipedia.org/wiki/Python",
        "RelatedTopics": [
            {"Text": "CPython - reference implementation", "FirstURL": "u1"},
            {"Topics": [{"Text": "PyPy - JIT compiler", "FirstURL": "u2"}]},
        ],
    }

    text = WebSearchTool.format_duckduckgo(data, count=5)

    assert "Python is a programming language." in text
    assert "CPython" in text and "PyPy" in text


def test_format_duckduckgo_empty_mentions_brave() -> None:
    text = WebSearchTool.format_duckduckgo({}, count=5)
    assert "BRAVE_SEARCH_API_KEY" in text


async def test_web_search_requires_query() -> None:
    tool = WebSearchTool()
    with pytest.raises(ToolError, match="query"):
        await tool.run({"query": "   "})
