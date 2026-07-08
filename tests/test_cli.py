"""Tests for the CLI REPL's answer path (input loop is not driven)."""

from __future__ import annotations

import pytest

from conftest import FakeLLM
from jarvis.brain.llm import LLMError
from jarvis.brain.orchestrator import Brain
from jarvis.interfaces.cli import CliRepl
from jarvis.memory.short_term import ConversationBuffer


def _repl(llm: FakeLLM) -> CliRepl:
    brain = Brain(
        llm=llm,
        history=ConversationBuffer(max_messages=10),
        system_prompt="test",
    )
    return CliRepl(brain, assistant_name="Jarvis")


async def test_answer_streams_reply(capsys: pytest.CaptureFixture[str]) -> None:
    repl = _repl(FakeLLM(["Hello there."]))

    await repl._answer("hi")

    out = capsys.readouterr().out
    assert "jarvis> " in out
    assert "Hello there." in out


async def test_answer_reports_brain_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repl = _repl(FakeLLM(error=LLMError("down")))

    await repl._answer("hi")  # must not raise

    out = capsys.readouterr().out
    assert "[error]" in out
    assert "language model" in out
