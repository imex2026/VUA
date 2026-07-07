"""Text-to-speech behind a swappable protocol.

Piper is the local default (fast, offline, per-language voices).
ElevenLabs is an opt-in cloud backend; when it is unreachable the voice
pipeline falls back to Piper and announces the switch aloud.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import structlog

from jarvis.audio.types import AudioClip

__all__ = ["ElevenLabsTts", "PiperTts", "TtsBackend", "TtsError"]


class TtsError(RuntimeError):
    """Synthesis failed (voice missing, network down, quota, ...)."""


class TtsBackend(Protocol):
    """Anything that can turn text into speech."""

    @property
    def name(self) -> str:
        """Short backend identifier for logs and announcements."""
        ...

    async def synthesize(self, text: str, language: str) -> AudioClip:
        """Render ``text`` (in ``language``) as int16 mono PCM."""
        ...


class PiperTts:
    """Local synthesis with Piper voices.

    ``voices`` maps a language code to a ``.onnx`` voice file (download
    from https://huggingface.co/rhasspy/piper-voices). Unknown
    languages fall back to ``default_language``. Voices load lazily and
    are cached; synthesis runs in a worker thread.
    """

    name = "piper"

    def __init__(
        self,
        *,
        voices: dict[str, str],
        default_language: str = "en",
    ) -> None:
        if not voices:
            raise TtsError(
                "No Piper voices configured. Set speech.tts.piper.voices "
                "in config.yaml (language -> path to a .onnx voice)."
            )
        if default_language not in voices:
            default_language = next(iter(voices))
        self._voice_paths = voices
        self._default_language = default_language
        self._loaded: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._log = structlog.get_logger("jarvis.speech.tts")

    def _load_voice(self, language: str) -> Any:
        path = self._voice_paths.get(
            language, self._voice_paths[self._default_language]
        )
        if path not in self._loaded:
            try:
                from piper import PiperVoice
            except ImportError as exc:  # pragma: no cover - env dependent
                raise TtsError(
                    "piper-tts is not installed. " "Run: pip install -e '.[audio]'"
                ) from exc
            try:
                self._loaded[path] = PiperVoice.load(path)
            except Exception as exc:
                raise TtsError(f"Could not load Piper voice {path}: {exc}") from exc
            self._log.info("tts_voice_loaded", path=path, language=language)
        return self._loaded[path]

    async def synthesize(self, text: str, language: str) -> AudioClip:
        """Render text with the voice configured for ``language``."""

        def _run() -> AudioClip:
            voice = self._load_voice(language)
            try:
                chunks: list[bytes] = []
                sample_rate = int(voice.config.sample_rate)
                if hasattr(voice, "synthesize_stream_raw"):
                    # piper-tts <= 1.2 streaming API
                    for raw in voice.synthesize_stream_raw(text):
                        chunks.append(raw)
                else:  # piper-tts >= 1.3 chunk objects
                    for chunk in voice.synthesize(text):
                        chunks.append(chunk.audio_int16_bytes)
                        sample_rate = int(chunk.sample_rate)
                return AudioClip(pcm=b"".join(chunks), sample_rate=sample_rate)
            except Exception as exc:
                raise TtsError(f"Piper synthesis failed: {exc}") from exc

        async with self._lock:
            return await asyncio.to_thread(_run)


class ElevenLabsTts:
    """Cloud synthesis via the ElevenLabs HTTP API.

    Requests raw PCM output so no audio decoding is needed. Any
    network or API failure raises :class:`TtsError`, which the voice
    pipeline treats as the signal to degrade to the local backend.
    """

    name = "elevenlabs"

    _SAMPLE_RATE = 22050

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_multilingual_v2",
        timeout_s: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._timeout_s = timeout_s
        self._log = structlog.get_logger("jarvis.speech.tts")

    async def synthesize(self, text: str, language: str) -> AudioClip:
        """POST to the ElevenLabs API and return the PCM response."""
        import httpx  # dependency of the anthropic SDK, always present

        url = (
            "https://api.elevenlabs.io/v1/text-to-speech/"
            f"{self._voice_id}?output_format=pcm_{self._SAMPLE_RATE}"
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(
                    url,
                    headers={"xi-api-key": self._api_key},
                    json={"text": text, "model_id": self._model_id},
                )
        except httpx.HTTPError as exc:
            raise TtsError(f"ElevenLabs request failed: {exc}") from exc
        if response.status_code != 200:
            raise TtsError(
                f"ElevenLabs returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        return AudioClip(pcm=response.content, sample_rate=self._SAMPLE_RATE)
