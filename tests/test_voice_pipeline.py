"""Integration-style tests for the voice pipeline with all backends faked."""

from __future__ import annotations

from conftest import (
    SILENCE,
    SPEECH,
    WAKE,
    FakeAudioSource,
    FakeLLM,
    FakeSink,
    FakeStt,
    FakeTts,
    FakeVad,
    FakeWake,
)
from jarvis.audio.endpointing import UtteranceCollector
from jarvis.brain.orchestrator import Brain
from jarvis.events import EventBus, TranscriptReady, WakeDetected
from jarvis.interfaces.voice import VoicePipeline
from jarvis.memory.short_term import ConversationBuffer

ONE_TURN = [SILENCE, WAKE] + [SPEECH] * 4 + [SILENCE] * 3


def _collector() -> UtteranceCollector:
    return UtteranceCollector(
        FakeVad(),
        speech_threshold=0.5,
        silence_ms=240.0,
        no_speech_timeout_ms=400.0,
        max_utterance_ms=1600.0,
        min_speech_ms=160.0,
    )


def _pipeline(
    payloads: list[bytes],
    *,
    llm: FakeLLM | None = None,
    tts: FakeTts | None = None,
    tts_fallback: FakeTts | None = None,
) -> tuple[VoicePipeline, FakeStt, FakeSink, EventBus]:
    bus = EventBus()
    stt = FakeStt()
    sink = FakeSink()
    brain = Brain(
        llm=llm or FakeLLM(["Lights are on."]),
        history=ConversationBuffer(max_messages=20),
        system_prompt="test",
        bus=bus,
    )
    pipeline = VoicePipeline(
        source=FakeAudioSource(payloads),
        wake=FakeWake(),
        collector=_collector(),
        stt=stt,
        tts=tts or FakeTts(),
        tts_fallback=tts_fallback,
        sink=sink,
        brain=brain,
        bus=bus,
    )
    return pipeline, stt, sink, bus


async def test_full_turn_wake_to_speech() -> None:
    pipeline, stt, sink, bus = _pipeline(ONE_TURN)
    events: list[str] = []

    async def on_wake(event: WakeDetected) -> None:
        events.append(f"wake:{event.wake_word}")

    async def on_transcript(event: TranscriptReady) -> None:
        events.append(f"transcript:{event.text}")

    bus.subscribe(WakeDetected, on_wake)
    bus.subscribe(TranscriptReady, on_transcript)

    await pipeline.run()
    await bus.join()

    # STT received exactly the recorded utterance (4 speech + 3 silence).
    assert len(stt.clips) == 1
    assert len(stt.clips[0].pcm) == 7 * len(SPEECH)
    # The reply was spoken.
    assert [clip.pcm for clip in sink.played] == [b"Lights are on."]
    assert "wake:hey_jarvis" in events
    assert "transcript:turn on the lights" in events


async def test_no_wake_no_transcription() -> None:
    pipeline, stt, sink, _ = _pipeline([SILENCE, SPEECH, SPEECH, SILENCE])

    await pipeline.run()

    assert stt.clips == []
    assert sink.played == []


async def test_wake_without_speech_returns_to_idle() -> None:
    # Wake, then 5 silent frames exhaust the no-speech timeout; a later
    # wake + speech still works.
    payloads = [WAKE] + [SILENCE] * 5 + ONE_TURN
    pipeline, stt, sink, _ = _pipeline(payloads)

    await pipeline.run()

    assert len(stt.clips) == 1
    assert len(sink.played) == 1


async def test_tts_language_follows_transcript() -> None:
    tts = FakeTts()
    pipeline, stt, _, _ = _pipeline(ONE_TURN, tts=tts)
    stt.language = "fr"

    await pipeline.run()

    assert tts.requests == [("Lights are on.", "fr")]


async def test_cloud_tts_failure_degrades_and_announces() -> None:
    cloud = FakeTts(name="cloud", fail=True)
    local = FakeTts(name="local")
    pipeline, _, sink, _ = _pipeline(ONE_TURN, tts=cloud, tts_fallback=local)

    await pipeline.run()

    # The notice was spoken first, then the actual reply, both locally.
    assert len(local.requests) == 2
    assert "local one" in local.requests[0][0]
    assert local.requests[1][0] == "Lights are on."
    assert len(sink.played) == 2


async def test_tts_failure_without_fallback_stays_silent() -> None:
    broken = FakeTts(fail=True)
    pipeline, _, sink, _ = _pipeline(ONE_TURN, tts=broken)

    await pipeline.run()  # must not raise

    assert sink.played == []
