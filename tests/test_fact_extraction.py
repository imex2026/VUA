"""Unit tests for LLM-based fact extraction."""

from __future__ import annotations

from conftest import FakeLLM
from jarvis.memory.extraction import FactExtractor


async def test_extracts_line_separated_facts() -> None:
    llm = FakeLLM(["The user's name is Aymen.\nThe user prefers replies in French."])
    extractor = FactExtractor(llm)

    facts = await extractor.extract("je m'appelle Aymen", "Enchante, Aymen!")

    assert facts == [
        "The user's name is Aymen.",
        "The user prefers replies in French.",
    ]
    # The turn was passed to the model.
    prompt = llm.calls[0]["messages"][0].content
    assert "je m'appelle Aymen" in prompt
    assert "Enchante, Aymen!" in prompt


async def test_none_output_means_no_facts() -> None:
    extractor = FactExtractor(FakeLLM(["NONE"]))

    assert await extractor.extract("what time is it?", "It's noon.") == []


def test_parse_strips_bullets_and_blanks() -> None:
    output = "- fact one\n\n* fact two\n  \n• fact three\nNONE\n"

    assert FactExtractor.parse(output, max_facts=5) == [
        "fact one",
        "fact two",
        "fact three",
    ]


def test_parse_caps_at_max_facts() -> None:
    output = "\n".join(f"fact {i}" for i in range(10))

    assert len(FactExtractor.parse(output, max_facts=3)) == 3
