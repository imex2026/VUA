"""Web search tool.

Uses the Brave Search API when a key is configured; otherwise falls
back to DuckDuckGo's keyless instant-answer API (shallower results,
but zero setup and no account).
"""

from __future__ import annotations

from typing import Any

import httpx

from jarvis.tools.base import ToolError

__all__ = ["WebSearchTool"]


class WebSearchTool:
    """Search the web and return the top results as text."""

    name = "web_search"
    description = (
        "Search the web for current information. Returns the top "
        "results with title, URL, and snippet. Use for facts you do "
        "not know or that may have changed recently."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "count": {
                "type": "integer",
                "description": "Number of results (1-10, default 5).",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self, *, brave_api_key: str | None = None, timeout_s: float = 10.0
    ) -> None:
        self._brave_api_key = brave_api_key
        self._timeout_s = timeout_s

    async def run(self, arguments: dict[str, Any]) -> str:
        """Run one search; provider depends on configured keys."""
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ToolError("web_search requires a non-empty 'query'.")
        count = max(1, min(10, int(arguments.get("count", 5))))
        if self._brave_api_key:
            return await self._search_brave(query, count)
        return await self._search_duckduckgo(query, count)

    async def _search_brave(self, query: str, count: int) -> str:
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": count},
                    headers={
                        "X-Subscription-Token": self._brave_api_key or "",
                        "Accept": "application/json",
                    },
                )
        except httpx.HTTPError as exc:
            raise ToolError(f"Web search failed: {exc}") from exc
        if response.status_code != 200:
            raise ToolError(f"Brave search returned HTTP {response.status_code}.")
        return self.format_brave(response.json(), count)

    async def _search_duckduckgo(self, query: str, count: int) -> str:
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": 1,
                        "skip_disambig": 1,
                    },
                )
        except httpx.HTTPError as exc:
            raise ToolError(f"Web search failed: {exc}") from exc
        if response.status_code != 200:
            raise ToolError(f"DuckDuckGo returned HTTP {response.status_code}.")
        return self.format_duckduckgo(response.json(), count)

    @staticmethod
    def format_brave(data: dict[str, Any], count: int) -> str:
        """Render a Brave API payload as numbered results."""
        results = (data.get("web") or {}).get("results") or []
        lines: list[str] = []
        for index, item in enumerate(results[:count], start=1):
            title = item.get("title", "(untitled)")
            url = item.get("url", "")
            snippet = item.get("description", "")
            lines.append(f"{index}. {title}\n   {url}\n   {snippet}")
        if not lines:
            return "No results found."
        return "\n".join(lines)

    @staticmethod
    def format_duckduckgo(data: dict[str, Any], count: int) -> str:
        """Render a DuckDuckGo instant-answer payload as text."""
        lines: list[str] = []
        abstract = data.get("AbstractText") or ""
        if abstract:
            source = data.get("AbstractURL") or data.get("AbstractSource") or ""
            lines.append(f"Summary: {abstract}\n   {source}")
        related = data.get("RelatedTopics") or []
        for topic in related:
            if len(lines) >= count:
                break
            # Grouped topics nest one level deeper.
            entries = topic.get("Topics", [topic])
            for entry in entries:
                if len(lines) >= count:
                    break
                text = entry.get("Text")
                url = entry.get("FirstURL", "")
                if text:
                    lines.append(f"- {text}\n   {url}")
        if not lines:
            return (
                "No results found. (Keyless DuckDuckGo mode is shallow; "
                "configure BRAVE_SEARCH_API_KEY for full web search.)"
            )
        return "\n".join(lines)
