"""Conversation orchestration between interfaces, memory, and the LLM.

Implements Claude's native tool-use loop: the model streams text and
may request tool calls; results are appended to history and the model
is called again until it produces a final answer (or the iteration
cap is hit).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator

import structlog

from jarvis.brain.llm import (
    ChatMessage,
    LLMBackend,
    LLMError,
    ResponseComplete,
    TextDelta,
    ToolUseRequest,
)
from jarvis.events import (
    EventBus,
    SpeechOutput,
    ToolCallCompleted,
    ToolCallRequested,
)
from jarvis.memory.short_term import ConversationBuffer
from jarvis.tools.base import ToolRegistry

if TYPE_CHECKING:
    from jarvis.memory.extraction import FactExtractor
    from jarvis.memory.long_term import LongTermMemory

__all__ = ["Brain", "BrainError"]

_TOOL_LIMIT_NOTE = (
    "I had to stop there - that request needed more tool calls than "
    "I allow in one turn."
)


class BrainError(RuntimeError):
    """Raised when a reply could not be produced.

    ``str(exc)`` is safe to show (or speak) to the user; the underlying
    cause is chained for the logs.
    """


class Brain:
    """Turns user text into streamed assistant replies, running tools
    as the model requests them.

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
        tools: ToolRegistry | None = None,
        max_tool_iterations: int = 8,
        memory: "LongTermMemory | None" = None,
        extractor: "FactExtractor | None" = None,
    ) -> None:
        self._llm = llm
        self._history = history
        self._system_prompt = system_prompt
        self._bus = bus
        self._tools = tools
        self._max_tool_iterations = max_tool_iterations
        self._memory = memory
        self._extractor = extractor
        self._background: set[asyncio.Task[None]] = set()
        self._log = structlog.get_logger("jarvis.brain")

    async def respond(self, user_text: str) -> AsyncIterator[str]:
        """Stream the assistant's reply to ``user_text``.

        Text is yielded as it streams, across tool-use rounds. On LLM
        failure the history is rolled back to the state before this
        turn, and :class:`BrainError` is raised with a
        user-presentable message.
        """
        system_prompt = await self._compose_system_prompt(user_text)
        checkpoint = len(self._history)
        self._history.append(ChatMessage(role="user", content=user_text))
        self._log.info("user_message", chars=len(user_text))

        specs = self._tools.specs() if self._tools is not None else []
        all_parts: list[str] = []
        iterations = 0
        try:
            while True:
                round_parts: list[str] = []
                requests: list[ToolUseRequest] = []
                async for event in self._llm.stream(
                    system=system_prompt,
                    messages=self._history.messages(),
                    tools=specs or None,
                ):
                    match event:
                        case TextDelta(text=text):
                            round_parts.append(text)
                            all_parts.append(text)
                            yield text
                        case ToolUseRequest() as request:
                            requests.append(request)
                        case ResponseComplete() as done:
                            self._log.info(
                                "llm_response_complete",
                                stop_reason=done.stop_reason,
                                input_tokens=done.input_tokens,
                                output_tokens=done.output_tokens,
                                tool_calls=len(requests),
                            )

                if not requests:
                    reply = "".join(round_parts)
                    if reply:
                        self._history.append(
                            ChatMessage(role="assistant", content=reply)
                        )
                    break

                self._history.append(
                    ChatMessage(
                        role="assistant",
                        content=self._assistant_blocks(round_parts, requests),
                    )
                )
                self._history.append(
                    ChatMessage(
                        role="user",
                        content=await self._run_tools(requests),
                    )
                )
                iterations += 1
                if iterations >= self._max_tool_iterations:
                    self._log.warning(
                        "tool_iteration_limit", limit=self._max_tool_iterations
                    )
                    all_parts.append(_TOOL_LIMIT_NOTE)
                    yield _TOOL_LIMIT_NOTE
                    self._history.append(
                        ChatMessage(role="assistant", content=_TOOL_LIMIT_NOTE)
                    )
                    break
        except LLMError as exc:
            self._history.truncate(checkpoint)
            self._log.error("llm_request_failed", error=str(exc))
            raise BrainError(
                "I couldn't reach the language model. "
                "Check the network and API key, then try again."
            ) from exc

        full = "".join(all_parts)
        if full and self._bus is not None:
            self._bus.publish(SpeechOutput(text=full))
        if full and self._memory is not None and self._extractor is not None:
            task = asyncio.create_task(self._memorize(user_text, full))
            self._background.add(task)
            task.add_done_callback(self._background.discard)

    @staticmethod
    def _assistant_blocks(
        parts: list[str], requests: list[ToolUseRequest]
    ) -> list[dict[str, Any]]:
        """Assistant content blocks for a round that requested tools."""
        blocks: list[dict[str, Any]] = []
        text = "".join(parts)
        if text:
            blocks.append({"type": "text", "text": text})
        for request in requests:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": request.id,
                    "name": request.name,
                    "input": request.arguments,
                }
            )
        return blocks

    async def _run_tools(self, requests: list[ToolUseRequest]) -> list[dict[str, Any]]:
        """Execute requested tools; return tool_result content blocks."""
        assert self._tools is not None  # requests imply tools were sent
        blocks: list[dict[str, Any]] = []
        for request in requests:
            if self._bus is not None:
                self._bus.publish(
                    ToolCallRequested(
                        call_id=request.id,
                        tool_name=request.name,
                        arguments=request.arguments,
                    )
                )
            self._log.info("tool_call", tool=request.name)
            result = await self._tools.execute(request.name, request.arguments)
            if self._bus is not None:
                self._bus.publish(
                    ToolCallCompleted(
                        call_id=request.id,
                        tool_name=request.name,
                        result=result.content,
                        is_error=result.is_error,
                    )
                )
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": request.id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )
        return blocks

    async def _compose_system_prompt(self, user_text: str) -> str:
        """Base persona plus any relevant long-term memories.

        A memory failure must never block a reply, so recall errors are
        logged and the base prompt is used.
        """
        if self._memory is None:
            return self._system_prompt
        try:
            facts = await self._memory.recall(user_text)
        except Exception as exc:
            self._log.warning("memory_recall_failed", error=str(exc))
            return self._system_prompt
        if not facts:
            return self._system_prompt
        bullets = "\n".join(f"- {fact}" for fact in facts)
        return (
            self._system_prompt
            + "\n\nRelevant things you remember about the user from "
            + "earlier conversations:\n"
            + bullets
        )

    async def _memorize(self, user_text: str, reply: str) -> None:
        """Extract and store facts from a finished turn (background)."""
        assert self._memory is not None and self._extractor is not None
        try:
            facts = await self._extractor.extract(user_text, reply)
            if facts:
                added = await self._memory.remember(facts)
                self._log.info("memorized", extracted=len(facts), added=added)
        except Exception as exc:
            self._log.warning("memorize_failed", error=str(exc))

    async def join_background(self) -> None:
        """Wait for in-flight memorization tasks (shutdown and tests)."""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    def reset(self) -> None:
        """Start a fresh conversation."""
        self._history.clear()
        self._log.info("conversation_reset")
