"""Voice activity detection behind a swappable protocol.

Two backends: Silero VAD (accurate, needs the ``audio`` extra) and a
dependency-free energy gate useful on constrained machines and in
tests.
"""

from __future__ import annotations

import asyncio
import math
from array import array
from typing import Any, Protocol

from jarvis.audio.types import AudioError, AudioFrame

__all__ = ["EnergyVad", "SileroVad", "VoiceActivityDetector"]


class VoiceActivityDetector(Protocol):
    """Estimates the probability that a frame contains speech."""

    async def speech_probability(self, frame: AudioFrame) -> float:
        """Return a probability in [0, 1] for this frame."""
        ...

    def reset(self) -> None:
        """Clear internal state between utterances."""
        ...


class SileroVad:
    """Silero VAD via the official ``silero-vad`` package.

    The model scores 512-sample windows at 16 kHz; incoming frames are
    re-chunked internally and the frame's probability is the maximum
    over its windows. Inference runs in a worker thread.
    """

    _WINDOW = 512  # samples per model invocation at 16 kHz

    def __init__(self) -> None:
        self._model: Any = None
        self._pending = b""

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from silero_vad import load_silero_vad
            except ImportError as exc:  # pragma: no cover - env dependent
                raise AudioError(
                    "silero-vad is not installed. " "Run: pip install -e '.[audio]'"
                ) from exc
            self._model = load_silero_vad(onnx=True)
        return self._model

    async def speech_probability(self, frame: AudioFrame) -> float:
        """Max speech probability across the frame's 512-sample windows."""

        def _run() -> float:
            import numpy
            import torch

            model = self._ensure_model()
            data = self._pending + frame.pcm
            window_bytes = self._WINDOW * 2
            best = 0.0
            offset = 0
            while offset + window_bytes <= len(data):
                chunk = numpy.frombuffer(
                    data[offset : offset + window_bytes], numpy.int16
                )
                samples = torch.from_numpy(chunk.astype(numpy.float32) / 32768.0)
                best = max(best, float(model(samples, frame.sample_rate).item()))
                offset += window_bytes
            self._pending = data[offset:]
            return best

        return await asyncio.to_thread(_run)

    def reset(self) -> None:
        """Drop buffered samples and the model's recurrent state."""
        self._pending = b""
        if self._model is not None:
            self._model.reset_states()


class EnergyVad:
    """A pure-Python RMS energy gate. No dependencies, no accuracy
    guarantees - a pragmatic fallback and test double.

    ``threshold`` is RMS as a fraction of int16 full scale; typical
    speech at normal mic gain lands well above the 0.015 default.
    """

    def __init__(self, threshold: float = 0.015) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self._threshold = threshold

    async def speech_probability(self, frame: AudioFrame) -> float:
        """Map frame RMS to a probability: 0.5 at threshold, 1.0 at 2x."""
        samples = array("h")
        samples.frombytes(frame.pcm)
        if not samples:
            return 0.0
        rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0
        return min(1.0, 0.5 * rms / self._threshold)

    def reset(self) -> None:
        """Stateless; nothing to clear."""
