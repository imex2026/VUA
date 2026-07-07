"""Streaming microphone capture.

``sounddevice`` (PortAudio) works on both Linux and Windows and is
imported lazily so the package loads without the ``audio`` extra.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Protocol

import structlog

from jarvis.audio.types import AudioError, AudioFrame

__all__ = ["AudioSource", "MicrophoneSource"]


class AudioSource(Protocol):
    """Anything that produces a stream of audio frames."""

    def frames(self) -> AsyncIterator[AudioFrame]:
        """Yield frames until the source ends or is cancelled."""
        ...


class MicrophoneSource:
    """Captures fixed-size int16 mono frames from the default (or a
    configured) input device.

    The PortAudio callback runs on its own thread; frames hop onto the
    event loop through an ``asyncio.Queue``. When the consumer falls
    behind, the oldest frames are dropped rather than blocking capture.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 80,
        device: int | str | None = None,
        queue_size: int = 100,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_samples = sample_rate * frame_ms // 1000
        self._device = device
        self._queue_size = queue_size
        self._log = structlog.get_logger("jarvis.audio.capture")

    async def frames(self) -> AsyncIterator[AudioFrame]:
        """Open the input stream and yield frames forever."""
        try:
            import sounddevice
        except ImportError as exc:  # pragma: no cover - env dependent
            raise AudioError(
                "sounddevice is not installed. Run: pip install -e '.[audio]'"
            ) from exc

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=self._queue_size)

        def deliver(pcm: bytes) -> None:
            if queue.full():
                queue.get_nowait()  # drop the oldest frame
                self._log.warning("mic_queue_overflow")
            queue.put_nowait(pcm)

        def callback(indata: object, *_args: object) -> None:
            loop.call_soon_threadsafe(deliver, bytes(indata))  # type: ignore[call-overload] # noqa: E501

        try:
            stream = sounddevice.RawInputStream(
                samplerate=self._sample_rate,
                blocksize=self._frame_samples,
                device=self._device,
                channels=1,
                dtype="int16",
                callback=callback,
            )
        except Exception as exc:
            raise AudioError(f"Could not open microphone: {exc}") from exc

        with stream:
            self._log.info(
                "microphone_open",
                sample_rate=self._sample_rate,
                frame_samples=self._frame_samples,
                device=self._device,
            )
            while True:
                pcm = await queue.get()
                yield AudioFrame(pcm=pcm, sample_rate=self._sample_rate)
