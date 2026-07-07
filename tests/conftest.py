"""Shared test doubles. The LLM and audio layers are always mocked."""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from jarvis.brain.llm import (
    ChatMessage,
    LLMError,
    LLMEvent,
    ResponseComplete,
    TextDelta,
)


class FakeLLM:
    """An LLMBackend that replays scripted replies and records calls."""

    def __init__(
        self,
        replies: list[str] | None = None,
        *,
        error: LLMError | None = None,
        chunk_size: int = 4,
    ) -> None:
        self.replies = list(replies or [])
        self.error = error
        self.chunk_size = chunk_size
        self.calls: list[dict[str, Any]] = []

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[LLMEvent]:
        self.calls.append({"system": system, "messages": list(messages)})
        if self.error is not None:
            raise self.error
        reply = self.replies.pop(0) if self.replies else "ok"
        for start in range(0, len(reply), self.chunk_size):
            yield TextDelta(reply[start : start + self.chunk_size])
        yield ResponseComplete(
            stop_reason="end_turn",
            input_tokens=len(str(messages)),
            output_tokens=len(reply),
        )
