"""Unit tests for the typed event bus."""

from __future__ import annotations

import asyncio

from jarvis.events import Event, EventBus, SpeechOutput, WakeDetected


async def test_publish_reaches_subscriber() -> None:
    bus = EventBus()
    received: list[WakeDetected] = []

    async def handler(event: WakeDetected) -> None:
        received.append(event)

    bus.subscribe(WakeDetected, handler)
    bus.publish(WakeDetected(wake_word="hey jarvis", score=0.93))
    await bus.join()

    assert len(received) == 1
    assert received[0].wake_word == "hey jarvis"


async def test_multiple_subscribers_all_receive() -> None:
    bus = EventBus()
    hits: list[str] = []

    async def first(event: SpeechOutput) -> None:
        hits.append("first")

    async def second(event: SpeechOutput) -> None:
        hits.append("second")

    bus.subscribe(SpeechOutput, first)
    bus.subscribe(SpeechOutput, second)
    bus.publish(SpeechOutput(text="hello"))
    await bus.join()

    assert sorted(hits) == ["first", "second"]


async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    received: list[SpeechOutput] = []

    async def handler(event: SpeechOutput) -> None:
        received.append(event)

    unsubscribe = bus.subscribe(SpeechOutput, handler)
    unsubscribe()
    bus.publish(SpeechOutput(text="hello"))
    await bus.join()

    assert received == []


async def test_base_class_subscription_receives_subclasses() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def audit(event: Event) -> None:
        seen.append(type(event).__name__)

    bus.subscribe(Event, audit)
    bus.publish(WakeDetected(wake_word="hey jarvis", score=0.9))
    bus.publish(SpeechOutput(text="hi"))
    await bus.join()

    assert sorted(seen) == ["SpeechOutput", "WakeDetected"]


async def test_failing_handler_does_not_break_others() -> None:
    bus = EventBus()
    received: list[SpeechOutput] = []

    async def broken(event: SpeechOutput) -> None:
        raise RuntimeError("boom")

    async def healthy(event: SpeechOutput) -> None:
        received.append(event)

    bus.subscribe(SpeechOutput, broken)
    bus.subscribe(SpeechOutput, healthy)
    bus.publish(SpeechOutput(text="resilient"))
    await bus.join()

    assert len(received) == 1


async def test_join_waits_for_slow_handlers() -> None:
    bus = EventBus()
    done = asyncio.Event()

    async def slow(event: SpeechOutput) -> None:
        await asyncio.sleep(0.05)
        done.set()

    bus.subscribe(SpeechOutput, slow)
    bus.publish(SpeechOutput(text="patience"))
    await bus.join()

    assert done.is_set()


async def test_aclose_cancels_pending_handlers() -> None:
    bus = EventBus()
    started = asyncio.Event()

    async def hang(event: SpeechOutput) -> None:
        started.set()
        await asyncio.sleep(60)

    bus.subscribe(SpeechOutput, hang)
    bus.publish(SpeechOutput(text="stuck"))
    await started.wait()
    await bus.aclose()  # returns promptly instead of waiting 60s
