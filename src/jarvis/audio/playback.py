"""Speaker output behind a swappable protocol."""

from __future__ import annotations

import asyncio
from typing import Protocol

import structlog

from jarvis.audio.types import AudioClip, AudioError

__all__ = ["AudioSink", "SoundDeviceSink"]


class AudioSink(Protocol):
    """Anything that can play a clip to the user."""

    async def play(self, clip: AudioClip) -> None:
        """Play ``clip`` to completion."""
        ...


class SoundDeviceSink:
    """Plays int16 mono PCM through PortAudio.

    The blocking write runs in a worker thread, keeping the event loop
    responsive during playback.
    """

    def __init__(self, *, device: int | str | None = None) -> None:
        self._device = device
        self._log = structlog.get_logger("jarvis.audio.playback")

    async def play(self, clip: AudioClip) -> None:
        """Open an output stream at the clip's rate and play it."""

        def _run() -> None:
            try:
                import sounddevice
            except ImportError as exc:  # pragma: no cover - env dependent
                raise AudioError(
                    "sounddevice is not installed. " "Run: pip install -e '.[audio]'"
                ) from exc
            try:
                with sounddevice.RawOutputStream(
                    samplerate=clip.sample_rate,
                    channels=1,
                    dtype="int16",
                    device=self._device,
                ) as stream:
                    stream.write(clip.pcm)
            except AudioError:
                raise
            except Exception as exc:
                raise AudioError(f"Playback failed: {exc}") from exc

        self._log.debug("playback_start", duration_ms=int(clip.duration_ms))
        await asyncio.to_thread(_run)
