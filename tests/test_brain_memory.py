"""Unit tests for the brain's memory integration (recall + extraction)."""

from __future__ import annotations

from typing import Sequence

from conftest import FakeLLM
from jarvis.brain.orchestrator import Brain
from jarvis.memory.extraction import FactExtractor
from jarvis.memory.short_term import ConversationBuffer


class FakeMemory:
    """Stands in for LongTermMemory."""

    def __init__(self, facts: list[str] | None = None) -> None:
        self.facts = facts or []
        self.remembered: list[str] = []
        self.recall_queries: list[str] = []

    async def recall(self, query: str) -> list[str]:
        self.recall_queries.append(query)
        return self.facts

    async def remember(self, facts: Sequence[str]) -> int:
        self.remembered.extend(facts)
        return len(facts)


class BrokenMemory:
    async def recall(self, query: str) -> list[str]:
        raise RuntimeError("store is corrupt")

    async def remember(self, facts: Sequence[str]) -> int:
        raise RuntimeError("store is corrupt")


def _brain(
    llm: FakeLLM,
    memory: FakeMemory | BrokenMemory,
    extractor: FactExtractor | None = None,
) -> Brain:
    return Brain(
        llm=llm,
        history=ConversationBuffer(max_messages=20),
        system_prompt="You are a test assistant.",
        memory=memory,  # type: ignore[arg-type]
        extractor=extractor,
    )


async def test_recalled_facts_injected_into_system_prompt() -> None:
    llm = FakeLLM(["Your name is Aymen, of course."])
    memory = FakeMemory(facts=["The user's name is Aymen."])
    brain = _brain(llm, memory)

    async for _ in brain.respond("what's my name?"):
        pass

    assert memory.recall_queries == ["what's my name?"]
    system = llm.calls[0]["system"]
    assert system.startswith("You are a test assistant.")
    assert "The user's name is Aymen." in system


async def test_no_facts_leaves_prompt_untouched() -> None:
    llm = FakeLLM(["hi"])
    brain = _brain(llm, FakeMemory(facts=[]))

    async for _ in brain.respond("hello"):
        pass

    assert llm.calls[0]["system"] == "You are a test assistant."


async def test_turn_is_memorized_in_background() -> None:
    reply_llm = FakeLLM(["Noted - green tea it is."])
    extractor_llm = FakeLLM(["The user prefers green tea."])
    memory = FakeMemory()
    brain = _brain(reply_llm, memory, FactExtractor(extractor_llm))

    async for _ in brain.respond("I prefer green tea"):
        pass
    await brain.join_background()

    assert memory.remembered == ["The user prefers green tea."]
    # The extractor saw both sides of the turn.
    prompt = extractor_llm.calls[0]["messages"][0].content
    assert "I prefer green tea" in prompt
    assert "Noted - green tea it is." in prompt


async def test_no_extractor_means_no_memorization() -> None:
    memory = FakeMemory()
    brain = _brain(FakeLLM(["ok"]), memory, extractor=None)

    async for _ in brain.respond("hello"):
        pass
    await brain.join_background()

    assert memory.remembered == []


async def test_memory_failure_does_not_block_reply() -> None:
    llm = FakeLLM(["still works"])
    brain = _brain(llm, BrokenMemory(), FactExtractor(FakeLLM(["a fact"])))

    chunks = [c async for c in brain.respond("hello")]
    await brain.join_background()  # memorize failure must be swallowed too

    assert "".join(chunks) == "still works"
    assert llm.calls[0]["system"] == "You are a test assistant."
