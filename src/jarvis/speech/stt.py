"""Speech-to-text behind a swappable protocol.

Default backend is faster-whisper (CTranslate2): local, fast on CPU
with int8, faster still on CUDA, and it auto-detects the spoken
language - Arabic, French, English, and German all work out of the box.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from jarvis.audio.types import AudioClip

__all__ = ["FasterWhisperStt", "SttBackend", "SttError", "Transcript"]


class SttError(RuntimeError):
    """Transcription failed (model missing, inference error, ...)."""


@dataclass(frozen=True, slots=True)
class Transcript:
    """The text of an utterance plus the detected language."""

    text: str
    language: str
    confidence: float


class SttBackend(Protocol):
    """Anything that can turn audio into text."""

    async def transcribe(self, clip: AudioClip) -> Transcript:
        """Transcribe ``clip`` with automatic language detection."""
        ...


class FasterWhisperStt:
    """faster-whisper implementation.

    The model loads lazily on first use and inference runs in a worker
    thread (CTranslate2 releases the GIL). ``device='auto'`` picks CUDA
    when available and falls back to CPU.
    """

    def __init__(
        self,
        *,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        allowed_languages: list[str] | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._allowed = allowed_languages
        self._model: Any = None
        self._lock = asyncio.Lock()
        self._log = structlog.get_logger("jarvis.speech.stt")

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - env dependent
                raise SttError(
                    "faster-whisper is not installed. " "Run: pip install -e '.[audio]'"
                ) from exc
            self._log.info(
                "stt_model_loading",
                model=self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            self._log.info("stt_model_ready")
        return self._model

    async def transcribe(self, clip: AudioClip) -> Transcript:
        """Transcribe one utterance; language is auto-detected."""

        def _run() -> Transcript:
            import numpy

            model = self._ensure_model()
            audio = (
                numpy.frombuffer(clip.pcm, numpy.int16).astype(numpy.float32) / 32768.0
            )
            try:
                segments, info = model.transcribe(audio, language=None, beam_size=5)
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception as exc:
                raise SttError(f"Transcription failed: {exc}") from exc
            return Transcript(
                text=text,
                language=info.language,
                confidence=float(info.language_probability),
            )

        # Whisper models are not reentrant; serialize access.
        async with self._lock:
            transcript = await asyncio.to_thread(_run)

        if self._allowed and transcript.language not in self._allowed:
            self._log.warning(
                "stt_unexpected_language",
                language=transcript.language,
                allowed=self._allowed,
            )
        self._log.info(
            "transcript",
            chars=len(transcript.text),
            language=transcript.language,
            confidence=round(transcript.confidence, 3),
        )
        return transcript
