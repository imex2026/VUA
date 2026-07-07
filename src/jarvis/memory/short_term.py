"""Short-term conversation memory: a bounded message buffer."""

from __future__ import annotations

from jarvis.brain.llm import ChatMessage

__all__ = ["ConversationBuffer"]


class ConversationBuffer:
    """A FIFO-trimmed buffer of chat messages.

    When the buffer exceeds ``max_messages`` the oldest messages are
    dropped, then further trimmed so the history always starts with a
    user turn (the Anthropic API requires the first message to be from
    the user).
    """

    def __init__(self, max_messages: int = 80) -> None:
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        self._max_messages = max_messages
        self._messages: list[ChatMessage] = []

    def append(self, message: ChatMessage) -> None:
        """Add a message, trimming the oldest ones if over capacity."""
        self._messages.append(message)
        overflow = len(self._messages) - self._max_messages
        if overflow > 0:
            del self._messages[:overflow]
        while self._messages and self._messages[0].role != "user":
            del self._messages[0]

    def pop(self) -> ChatMessage | None:
        """Remove and return the newest message, or None when empty."""
        return self._messages.pop() if self._messages else None

    def messages(self) -> list[ChatMessage]:
        """Return a copy of the buffered messages, oldest first."""
        return list(self._messages)

    def clear(self) -> None:
        """Forget the whole conversation."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
