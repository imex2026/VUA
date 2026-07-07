"""Conversation orchestration between interfaces, memory, and the LLM.

The tool-use loop (Claude native tools + MCP) is added here in Phase 3;
Phase 1 covers streaming replies with conversation history.
"""

from __future__ import annotations

from typing import AsyncIterator

import structlog

from jarvis.brain.llm import (
    ChatMessage,
    LLMBackend,
    LLMError,
    ResponseComplete,
    TextDelta,
)
from jarvis.events import EventBus, SpeechOutput
from jarvis.memory.short_term import ConversationBuffer

__all__ = ["Brain", "BrainError"]


class BrainError(RuntimeError):
    """Raised when a reply could not be produced.

    ``str(exc)`` is safe to show (or speak) to the user; the underlying
    cause is chained for the logs.
    """


class Brain:
    """Turns user text into streamed assistant replies.

    One :class:`Brain` holds one conversation. All interfaces (CLI,
    voice, HTTP) share this class so behaviour is identical everywhere.
    """

    def __init__(
        self,
        *,
        llm: LLMBackend,
        history: ConversationBuffer,
        system_prompt: str,
        bus: EventBus | None = None,
    ) -> None:
        self._llm = llm
        self._history = history
        self._system_prompt = system_prompt
        self._bus = bus
        self._log = structlog.get_logger("jarvis.brain")

    async def respond(self, user_text: str) -> AsyncIterator[str]:
        """Stream the assistant's reply to ``user_text``.

        The user message is committed to history only once the model
        answers; on failure it is rolled back so a retry doesn't
        duplicate turns, and :class:`BrainError` is raised with a
        user-presentable message.
        """
        self._history.append(ChatMessage(role="user", content=user_text))
        self._log.info("user_message", chars=len(user_text))

        parts: list[str] = []
        try:
            async for event in self._llm.stream(
                system=self._system_prompt,
                messages=self._history.messages(),
            ):
                match event:
                    case TextDelta(text=text):
                        parts.append(text)
                        yield text
                    case ResponseComplete() as done:
                        self._log.info(
                            "llm_response_complete",
                            stop_reason=done.stop_reason,
                            input_tokens=done.input_tokens,
                            output_tokens=done.output_tokens,
                        )
        except LLMError as exc:
            self._history.pop()
            self._log.error("llm_request_failed", error=str(exc))
            raise BrainError(
                "I couldn't reach the language model. "
                "Check the network and API key, then try again."
            ) from exc

        reply = "".join(parts)
        if reply:
            self._history.append(ChatMessage(role="assistant", content=reply))
            if self._bus is not None:
                self._bus.publish(SpeechOutput(text=reply))

    def reset(self) -> None:
        """Start a fresh conversation."""
        self._history.clear()
        self._log.info("conversation_reset")
