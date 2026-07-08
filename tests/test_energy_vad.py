"""Unit tests for the dependency-free energy VAD."""

from __future__ import annotations

import math
from array import array

import pytest

from jarvis.audio.types import AudioFrame
from jarvis.audio.vad import EnergyVad


def _tone_frame(amplitude: int, samples: int = 1280) -> AudioFrame:
    """A sine wave at the given int16 amplitude."""
    wave = array(
        "h",
        (
            int(amplitude * math.sin(2 * math.pi * 220 * i / 16000))
            for i in range(samples)
        ),
    )
    return AudioFrame(pcm=wave.tobytes(), sample_rate=16000)


async def test_silence_scores_low() -> None:
    vad = EnergyVad(threshold=0.015)
    frame = AudioFrame(pcm=b"\x00" * 2560, sample_rate=16000)

    assert await vad.speech_probability(frame) == 0.0


async def test_loud_audio_scores_high() -> None:
    vad = EnergyVad(threshold=0.015)
    # amplitude 3277 = 0.1 full scale, RMS = 0.07 >> threshold
    assert await vad.speech_probability(_tone_frame(3277)) == 1.0


async def test_quiet_audio_scores_in_between() -> None:
    vad = EnergyVad(threshold=0.015)
    # RMS = 0.005 -> probability = 0.5 * 0.005 / 0.015 = 0.17
    probability = await vad.speech_probability(_tone_frame(232))

    assert 0.0 < probability < 0.5


async def test_empty_frame_is_silent() -> None:
    vad = EnergyVad()
    assert await vad.speech_probability(AudioFrame(pcm=b"", sample_rate=16000)) == 0.0


def test_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        EnergyVad(threshold=0.0)
