"""Unit tests for the brain orchestrator with a mocked LLM."""

from __future__ import annotations

import pytest

from conftest import FakeLLM
from jarvis.brain.llm import LLMError
from jarvis.brain.orchestrator import Brain, BrainError
from jarvis.events import EventBus, SpeechOutput
from jarvis.memory.short_term import ConversationBuffer

SYSTEM = "You are a test assistant."


def _brain(llm: FakeLLM, bus: EventBus | None = None) -> Brain:
    return Brain(
        llm=llm,
        history=ConversationBuffer(max_messages=20),
        system_prompt=SYSTEM,
        bus=bus,
    )


async def test_streams_reply_and_records_history() -> None:
    llm = FakeLLM(["Hello there, sir."])
    brain = _brain(llm)

    chunks = [chunk async for chunk in brain.respond("hello")]

    assert "".join(chunks) == "Hello there, sir."
    assert len(chunks) > 1  # actually streamed, not one blob

    call = llm.calls[0]
    assert call["system"] == SYSTEM
    assert [m.role for m in call["messages"]] == ["user"]


async def test_history_accumulates_across_turns() -> None:
    llm = FakeLLM(["first", "second"])
    brain = _brain(llm)

    async for _ in brain.respond("one"):
        pass
    async for _ in brain.respond("two"):
        pass

    roles = [m.role for m in llm.calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]


async def test_llm_failure_raises_brain_error_and_rolls_back() -> None:
    llm = FakeLLM(error=LLMError("connection refused"))
    brain = _brain(llm)

    with pytest.raises(BrainError):
        async for _ in brain.respond("hello"):
            pass

    # The failed user turn was rolled back: a retry starts clean.
    llm.error = None
    llm.replies = ["recovered"]
    async for _ in brain.respond("hello again"):
        pass
    assert [m.role for m in llm.calls[1]["messages"]] == ["user"]


async def test_reply_published_as_speech_output() -> None:
    bus = EventBus()
    spoken: list[str] = []

    async def on_speech(event: SpeechOutput) -> None:
        spoken.append(event.text)

    bus.subscribe(SpeechOutput, on_speech)
    brain = _brain(FakeLLM(["as you wish"]), bus)

    async for _ in brain.respond("do the thing"):
        pass
    await bus.join()

    assert spoken == ["as you wish"]


async def test_reset_clears_history() -> None:
    llm = FakeLLM(["a", "b"])
    brain = _brain(llm)

    async for _ in brain.respond("one"):
        pass
    brain.reset()
    async for _ in brain.respond("two"):
        pass

    assert [m.role for m in llm.calls[1]["messages"]] == ["user"]
