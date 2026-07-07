"""Unit tests for the short-term conversation buffer."""

from __future__ import annotations

import pytest

from jarvis.brain.llm import ChatMessage
from jarvis.memory.short_term import ConversationBuffer


def _turn(index: int) -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content=f"question {index}"),
        ChatMessage(role="assistant", content=f"answer {index}"),
    ]


def test_append_and_read_back() -> None:
    buffer = ConversationBuffer(max_messages=10)
    for message in _turn(1):
        buffer.append(message)

    assert len(buffer) == 2
    assert [m.role for m in buffer.messages()] == ["user", "assistant"]


def test_trims_oldest_when_full() -> None:
    buffer = ConversationBuffer(max_messages=4)
    for index in range(1, 5):  # 8 messages through a 4-slot buffer
        for message in _turn(index):
            buffer.append(message)

    kept = buffer.messages()
    assert len(kept) == 4
    assert kept[0].content == "question 3"
    assert kept[0].role == "user"


def test_trim_keeps_user_message_first() -> None:
    buffer = ConversationBuffer(max_messages=3)
    for index in range(1, 4):
        for message in _turn(index):
            buffer.append(message)

    kept = buffer.messages()
    assert kept[0].role == "user"
    assert len(kept) <= 3


def test_messages_returns_a_copy() -> None:
    buffer = ConversationBuffer(max_messages=10)
    buffer.append(ChatMessage(role="user", content="hi"))
    snapshot = buffer.messages()
    snapshot.clear()

    assert len(buffer) == 1


def test_pop_and_clear() -> None:
    buffer = ConversationBuffer(max_messages=10)
    assert buffer.pop() is None

    buffer.append(ChatMessage(role="user", content="hi"))
    popped = buffer.pop()
    assert popped is not None and popped.content == "hi"

    buffer.append(ChatMessage(role="user", content="again"))
    buffer.clear()
    assert len(buffer) == 0


def test_rejects_tiny_capacity() -> None:
    with pytest.raises(ValueError):
        ConversationBuffer(max_messages=1)
