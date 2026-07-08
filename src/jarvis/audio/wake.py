"""Wake-word detection behind a swappable protocol.

Default backend is openWakeWord: fully local, no account required, and
ships a pretrained "hey jarvis" model. The ONNX inference framework is
used explicitly because the tflite runtime is unavailable on Windows.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from jarvis.audio.types import AudioError, AudioFrame

__all__ = ["OpenWakeWordDetector", "WakeDetection", "WakeWordDetector"]


@dataclass(frozen=True, slots=True)
class WakeDetection:
    """A wake word was heard."""

    wake_word: str
    score: float


class WakeWordDetector(Protocol):
    """Consumes frames; reports when the wake phrase is heard."""

    async def process(self, frame: AudioFrame) -> WakeDetection | None:
        """Return a detection if this frame triggered the wake word."""
        ...

    def reset(self) -> None:
        """Clear internal buffers (e.g. after an interaction)."""
        ...


class OpenWakeWordDetector:
    """openWakeWord-based detector.

    Model inference runs in a worker thread so the 80 ms frame cadence
    never blocks the event loop. A refractory period suppresses
    re-triggers while an interaction is already in flight.
    """

    def __init__(
        self,
        *,
        model: str = "hey_jarvis",
        threshold: float = 0.6,
        refractory_s: float = 2.0,
    ) -> None:
        self._model_name = model
        self._threshold = threshold
        self._refractory_s = refractory_s
        self._model: Any = None
        self._last_trigger = 0.0
        self._log = structlog.get_logger("jarvis.audio.wake")

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                import openwakeword
                from openwakeword.model import Model
            except ImportError as exc:  # pragma: no cover - env dependent
                raise AudioError(
                    "openwakeword is not installed. " "Run: pip install -e '.[audio]'"
                ) from exc
            # Fetches the shared feature models (and ours) on first run;
            # existing files are kept.
            openwakeword.utils.download_models([self._model_name])
            self._model = Model(
                wakeword_models=[self._model_name],
                inference_framework="onnx",
            )
            self._log.info("wake_model_loaded", model=self._model_name)
        return self._model

    async def process(self, frame: AudioFrame) -> WakeDetection | None:
        """Score one frame; return a detection above the threshold."""

        def _predict() -> float:
            import numpy

            model = self._ensure_model()
            scores = model.predict(numpy.frombuffer(frame.pcm, numpy.int16))
            return float(max(scores.values()))

        score = await asyncio.to_thread(_predict)
        now = time.monotonic()
        if score >= self._threshold and now - self._last_trigger >= self._refractory_s:
            self._last_trigger = now
            return WakeDetection(wake_word=self._model_name, score=score)
        return None

    def reset(self) -> None:
        """Clear the model's rolling audio buffer."""
        if self._model is not None:
            self._model.reset()
