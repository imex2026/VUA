"""Utterance endpointing: deciding when the user finished speaking.

Pure logic over VAD probabilities, so it is fully unit-testable with
no audio stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.audio.types import AudioFrame
from jarvis.audio.vad import VoiceActivityDetector

__all__ = ["NoSpeech", "Utterance", "UtteranceCollector"]


@dataclass(frozen=True, slots=True)
class Utterance:
    """A complete spoken utterance ready for transcription."""

    pcm: bytes
    sample_rate: int


@dataclass(frozen=True, slots=True)
class NoSpeech:
    """The user said nothing usable (timeout or too-short blip)."""

    reason: str


class UtteranceCollector:
    """State machine that turns frames + VAD scores into utterances.

    After the wake word, frames are fed here. The collector waits for
    speech to start (giving up after ``no_speech_timeout_ms``), then
    records until ``silence_ms`` of trailing quiet or the
    ``max_utterance_ms`` cap, and finally rejects blips shorter than
    ``min_speech_ms``.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        *,
        speech_threshold: float = 0.5,
        silence_ms: float = 800.0,
        no_speech_timeout_ms: float = 6000.0,
        max_utterance_ms: float = 30000.0,
        min_speech_ms: float = 300.0,
    ) -> None:
        self._vad = vad
        self._speech_threshold = speech_threshold
        self._silence_ms = silence_ms
        self._no_speech_timeout_ms = no_speech_timeout_ms
        self._max_utterance_ms = max_utterance_ms
        self._min_speech_ms = min_speech_ms
        self._chunks: list[bytes] = []
        self._elapsed_ms = 0.0
        self._speech_ms = 0.0
        self._trailing_silence_ms = 0.0
        self._speech_started = False

    def reset(self) -> None:
        """Prepare for a fresh utterance."""
        self._chunks = []
        self._elapsed_ms = 0.0
        self._speech_ms = 0.0
        self._trailing_silence_ms = 0.0
        self._speech_started = False
        self._vad.reset()

    async def feed(self, frame: AudioFrame) -> Utterance | NoSpeech | None:
        """Consume one frame.

        Returns ``None`` while listening continues, an
        :class:`Utterance` when the user finished speaking, or
        :class:`NoSpeech` when the attempt should be abandoned.
        """
        probability = await self._vad.speech_probability(frame)
        is_speech = probability >= self._speech_threshold
        duration = frame.duration_ms
        self._elapsed_ms += duration

        if not self._speech_started:
            if is_speech:
                self._speech_started = True
                self._chunks.append(frame.pcm)
                self._speech_ms += duration
            elif self._elapsed_ms >= self._no_speech_timeout_ms:
                self.reset()
                return NoSpeech(reason="timeout")
            return None

        self._chunks.append(frame.pcm)
        if is_speech:
            self._speech_ms += duration
            self._trailing_silence_ms = 0.0
        else:
            self._trailing_silence_ms += duration

        ended = (
            self._trailing_silence_ms >= self._silence_ms
            or self._elapsed_ms >= self._max_utterance_ms
        )
        if not ended:
            return None

        pcm = b"".join(self._chunks)
        speech_ms = self._speech_ms
        self.reset()
        if speech_ms < self._min_speech_ms:
            return NoSpeech(reason="too_short")
        return Utterance(pcm=pcm, sample_rate=frame.sample_rate)
