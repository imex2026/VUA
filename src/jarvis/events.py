"""Typed events and the asynchronous event bus connecting Jarvis's layers.

Every cross-layer signal travels as a frozen dataclass derived from
:class:`Event`, carrying a ``trace_id`` so a whole conversation can be
followed through the logs. The :class:`EventBus` dispatches events to
async subscribers as fire-and-forget tasks; a failing handler is logged
and never takes down the publisher or its sibling handlers.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, TypeVar

import structlog

__all__ = [
    "Event",
    "EventBus",
    "ShutdownRequested",
    "SpeechOutput",
    "ToolCallCompleted",
    "ToolCallRequested",
    "TranscriptReady",
    "UtteranceCaptured",
    "WakeDetected",
    "new_trace_id",
]


def new_trace_id() -> str:
    """Return a short random identifier used to correlate log lines."""
    return uuid.uuid4().hex[:12]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Base class for all bus events."""

    trace_id: str = field(default_factory=new_trace_id)
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True, slots=True, kw_only=True)
class WakeDetected(Event):
    """The wake-word engine heard the trigger phrase."""

    wake_word: str
    score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class UtteranceCaptured(Event):
    """VAD delimited a complete user utterance from the mic stream."""

    audio: bytes
    sample_rate: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptReady(Event):
    """STT produced text (with detected language) for an utterance."""

    text: str
    language: str
    source: str = "voice"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallRequested(Event):
    """The brain asked for a tool to be executed."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallCompleted(Event):
    """A tool finished; ``result`` is what goes back to the model."""

    call_id: str
    tool_name: str
    result: str
    is_error: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeechOutput(Event):
    """Final assistant text ready to be spoken and/or displayed."""

    text: str
    language: str = "en"


@dataclass(frozen=True, slots=True, kw_only=True)
class ShutdownRequested(Event):
    """Ask the application to wind down gracefully."""

    reason: str = ""


EventT = TypeVar("EventT", bound=Event)

# Internal storage erases the concrete event type; ``subscribe`` keeps the
# public API precisely typed.
_Handler = Callable[[Any], Awaitable[None]]


class EventBus:
    """A minimal in-process async pub/sub hub.

    Subscribing to a base class receives all of its subclasses, so a
    logger can watch plain :class:`Event` while the TTS worker watches
    only :class:`SpeechOutput`.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[_Handler]] = defaultdict(list)
        self._tasks: set[asyncio.Task[None]] = set()
        self._log = structlog.get_logger("jarvis.events")

    def subscribe(
        self,
        event_type: type[EventT],
        handler: Callable[[EventT], Awaitable[None]],
    ) -> Callable[[], None]:
        """Register ``handler`` for ``event_type``; returns an unsubscriber."""
        self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

        return unsubscribe

    def publish(self, event: Event) -> None:
        """Dispatch ``event`` to all matching handlers as background tasks.

        Must be called from within a running event loop. Handlers
        subscribed to both a class and its ancestor receive the event
        once per subscription.
        """
        for klass in type(event).__mro__:
            if klass is object:
                continue
            for handler in list(self._handlers.get(klass, ())):
                task = asyncio.create_task(self._dispatch(handler, event))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    async def _dispatch(self, handler: _Handler, event: Event) -> None:
        try:
            await handler(event)
        except Exception:
            self._log.exception(
                "event_handler_failed",
                event=type(event).__name__,
                trace_id=event.trace_id,
            )

    async def join(self) -> None:
        """Wait until every in-flight handler task has finished."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def aclose(self) -> None:
        """Cancel outstanding handler tasks and wait for them to settle."""
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)
        self._handlers.clear()
