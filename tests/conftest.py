"""Shared test doubles. The LLM and audio layers are always mocked."""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from jarvis.audio.types import AudioClip, AudioFrame
from jarvis.audio.wake import WakeDetection
from jarvis.brain.llm import (
    ChatMessage,
    LLMError,
    LLMEvent,
    ResponseComplete,
    TextDelta,
)
from jarvis.speech.stt import Transcript
from jarvis.speech.tts import TtsError


class FakeLLM:
    """An LLMBackend that replays scripted replies and records calls.

    A reply may be a plain string (streamed as text deltas) or a list
    of :data:`LLMEvent` objects for scripting tool-use rounds.
    """

    def __init__(
        self,
        replies: list[str | list[LLMEvent]] | None = None,
        *,
        error: LLMError | None = None,
        chunk_size: int = 4,
    ) -> None:
        self.replies = list(replies or [])
        self.error = error
        self.chunk_size = chunk_size
        self.calls: list[dict[str, Any]] = []

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": list(tools) if tools is not None else None,
            }
        )
        if self.error is not None:
            raise self.error
        reply = self.replies.pop(0) if self.replies else "ok"
        if isinstance(reply, list):
            for event in reply:
                yield event
            return
        for start in range(0, len(reply), self.chunk_size):
            yield TextDelta(reply[start : start + self.chunk_size])
        yield ResponseComplete(
            stop_reason="end_turn",
            input_tokens=len(str(messages)),
            output_tokens=len(reply),
        )


SAMPLE_RATE = 16000
FRAME_MS = 80
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2

# Marker payloads: the fake wake/VAD backends key off the first byte.
SILENCE = b"\x00" * FRAME_BYTES
SPEECH = b"\x01" + b"\x00" * (FRAME_BYTES - 1)
WAKE = b"\x02" + b"\x00" * (FRAME_BYTES - 1)


class FakeAudioSource:
    """Yields scripted frames, constructing them lazily so their
    ``captured_at`` timestamps reflect when the pipeline consumes them."""

    def __init__(self, payloads: list[bytes]) -> None:
        self._payloads = payloads

    async def frames(self) -> AsyncIterator[AudioFrame]:
        for pcm in self._payloads:
            yield AudioFrame(pcm=pcm, sample_rate=SAMPLE_RATE)


class FakeWake:
    """Triggers on WAKE-marked frames."""

    def __init__(self) -> None:
        self.resets = 0

    async def process(self, frame: AudioFrame) -> WakeDetection | None:
        if frame.pcm[:1] == b"\x02":
            return WakeDetection(wake_word="hey_jarvis", score=0.91)
        return None

    def reset(self) -> None:
        self.resets += 1


class FakeVad:
    """Reports speech for SPEECH-marked frames."""

    def __init__(self) -> None:
        self.resets = 0

    async def speech_probability(self, frame: AudioFrame) -> float:
        return 1.0 if frame.pcm[:1] == b"\x01" else 0.0

    def reset(self) -> None:
        self.resets += 1


class FakeStt:
    """Returns a fixed transcript and records what it was given."""

    def __init__(self, text: str = "turn on the lights", language: str = "en") -> None:
        self.text = text
        self.language = language
        self.clips: list[AudioClip] = []

    async def transcribe(self, clip: AudioClip) -> Transcript:
        self.clips.append(clip)
        return Transcript(text=self.text, language=self.language, confidence=0.99)


class FakeTts:
    """Encodes the request into the clip so tests can assert on it."""

    def __init__(self, name: str = "fake-tts", fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.requests: list[tuple[str, str]] = []

    async def synthesize(self, text: str, language: str) -> AudioClip:
        self.requests.append((text, language))
        if self.fail:
            raise TtsError(f"{self.name} unavailable")
        return AudioClip(pcm=text.encode(), sample_rate=SAMPLE_RATE)


class FakeSink:
    """Records every clip played."""

    def __init__(self) -> None:
        self.played: list[AudioClip] = []

    async def play(self, clip: AudioClip) -> None:
        self.played.append(clip)
