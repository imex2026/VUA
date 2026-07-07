"""Unit tests for the utterance endpointing state machine."""

from __future__ import annotations

from conftest import SAMPLE_RATE, SILENCE, SPEECH, FakeVad
from jarvis.audio.endpointing import NoSpeech, Utterance, UtteranceCollector
from jarvis.audio.types import AudioFrame


def _frame(pcm: bytes) -> AudioFrame:
    return AudioFrame(pcm=pcm, sample_rate=SAMPLE_RATE)


def _collector(**overrides: float) -> UtteranceCollector:
    defaults: dict[str, float] = {
        "speech_threshold": 0.5,
        "silence_ms": 240.0,  # 3 silent frames at 80 ms
        "no_speech_timeout_ms": 400.0,  # 5 frames
        "max_utterance_ms": 1600.0,  # 20 frames
        "min_speech_ms": 160.0,  # 2 speech frames
    }
    defaults.update(overrides)
    return UtteranceCollector(FakeVad(), **defaults)


async def test_speech_then_silence_yields_utterance() -> None:
    collector = _collector()
    frames = [SPEECH] * 4 + [SILENCE] * 3
    results = [await collector.feed(_frame(p)) for p in frames]

    assert all(r is None for r in results[:-1])
    final = results[-1]
    assert isinstance(final, Utterance)
    # 4 speech + 3 silence frames were recorded
    assert len(final.pcm) == 7 * len(SPEECH)
    assert final.sample_rate == SAMPLE_RATE


async def test_leading_silence_is_not_recorded() -> None:
    collector = _collector()
    frames = [SILENCE, SILENCE, SPEECH, SPEECH, SPEECH] + [SILENCE] * 3
    results = [await collector.feed(_frame(p)) for p in frames]

    final = results[-1]
    assert isinstance(final, Utterance)
    assert len(final.pcm) == 6 * len(SPEECH)  # leading silence excluded


async def test_no_speech_timeout() -> None:
    collector = _collector()
    results = [await collector.feed(_frame(SILENCE)) for _ in range(5)]

    assert results[:-1] == [None] * 4
    final = results[-1]
    assert isinstance(final, NoSpeech)
    assert final.reason == "timeout"


async def test_short_blip_is_rejected() -> None:
    collector = _collector()
    frames = [SPEECH] + [SILENCE] * 3  # 80 ms of speech < 160 ms minimum
    results = [await collector.feed(_frame(p)) for p in frames]

    final = results[-1]
    assert isinstance(final, NoSpeech)
    assert final.reason == "too_short"


async def test_max_duration_forces_end() -> None:
    collector = _collector(max_utterance_ms=800.0)  # 10 frames
    result: object = None
    for index in range(10):
        result = await collector.feed(_frame(SPEECH))
        if index < 9:
            assert result is None

    assert isinstance(result, Utterance)
    assert len(result.pcm) == 10 * len(SPEECH)


async def test_reset_clears_progress() -> None:
    collector = _collector()
    await collector.feed(_frame(SPEECH))
    collector.reset()

    frames = [SPEECH, SPEECH] + [SILENCE] * 3
    results = [await collector.feed(_frame(p)) for p in frames]
    final = results[-1]
    assert isinstance(final, Utterance)
    assert len(final.pcm) == 5 * len(SPEECH)  # pre-reset frame is gone
