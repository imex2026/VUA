"""LLM backend abstraction and the Anthropic (Claude) implementation.

Backends implement :class:`LLMBackend`: an async generator that yields
:class:`TextDelta` chunks as the model streams and finishes with a
:class:`ResponseComplete`. Tool-use events join this union in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Protocol, Sequence

import anthropic

__all__ = [
    "AnthropicBackend",
    "ChatMessage",
    "LLMBackend",
    "LLMError",
    "LLMEvent",
    "ResponseComplete",
    "TextDelta",
    "ToolUseRequest",
]


class LLMError(RuntimeError):
    """A backend failed to produce a response (network, auth, quota...)."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in the conversation history.

    ``content`` is plain text for normal turns; tool results (Phase 3)
    use the block-list form.
    """

    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A streamed fragment of assistant text."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolUseRequest:
    """The model wants a tool executed before it can finish."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResponseComplete:
    """Terminal stream event carrying stop reason and token usage."""

    stop_reason: str | None
    input_tokens: int
    output_tokens: int


LLMEvent = TextDelta | ToolUseRequest | ResponseComplete


class LLMBackend(Protocol):
    """Anything that can stream a chat completion."""

    def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """Stream a reply to ``messages`` under the ``system`` prompt.

        ``tools`` are Anthropic-format tool specs; when the model calls
        one, :class:`ToolUseRequest` events are yielded before
        :class:`ResponseComplete`.
        """
        ...


class AnthropicBackend:
    """Claude via the official Anthropic SDK, with streaming."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """Yield text deltas (and tool-use requests), then completion."""
        payload = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = list(tools)
        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system,
                messages=payload,  # type: ignore[arg-type]
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield TextDelta(text)
                final = await stream.get_final_message()
            for block in final.content:
                if block.type == "tool_use":
                    yield ToolUseRequest(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input),
                    )
            yield ResponseComplete(
                stop_reason=final.stop_reason,
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
            )
        except anthropic.AnthropicError as exc:
            raise LLMError(str(exc)) from exc
