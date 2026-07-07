"""Shared audio value types: 16-bit mono PCM frames and clips."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

__all__ = ["AudioClip", "AudioError", "AudioFrame"]


class AudioError(RuntimeError):
    """An audio backend failed (device missing, driver error, ...)."""


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """A short chunk of microphone audio (int16 mono PCM).

    ``captured_at`` (monotonic clock) lets the pipeline discard frames
    that were recorded while Jarvis itself was speaking.
    """

    pcm: bytes
    sample_rate: int
    captured_at: float = field(default_factory=time.monotonic)

    @property
    def duration_ms(self) -> float:
        """Frame length in milliseconds."""
        return (len(self.pcm) / 2) / self.sample_rate * 1000.0


@dataclass(frozen=True, slots=True)
class AudioClip:
    """A complete utterance or synthesized speech (int16 mono PCM)."""

    pcm: bytes
    sample_rate: int

    @property
    def duration_ms(self) -> float:
        """Clip length in milliseconds."""
        return (len(self.pcm) / 2) / self.sample_rate * 1000.0
