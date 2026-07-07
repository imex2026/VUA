"""Composition root: wires configuration, logging, bus, brain, interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from jarvis.interfaces.voice import VoicePipeline
    from jarvis.speech.tts import TtsBackend

from jarvis.brain.llm import AnthropicBackend
from jarvis.brain.orchestrator import Brain
from jarvis.brain.persona import build_system_prompt
from jarvis.config import ConfigError, JarvisSettings, load_settings
from jarvis.events import EventBus
from jarvis.interfaces.cli import CliRepl
from jarvis.log import configure_logging
from jarvis.memory.short_term import ConversationBuffer

__all__ = ["JarvisApp"]


class JarvisApp:
    """Owns the long-lived pieces and runs one interface at a time."""

    def __init__(self, settings: JarvisSettings) -> None:
        configure_logging(settings.app.log_level, settings.app.log_format)
        self._settings = settings
        self._bus = EventBus()
        self._log = structlog.get_logger("jarvis.app")

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "JarvisApp":
        """Build an app from ``config.yaml`` (or ``config_path``)."""
        return cls(load_settings(config_path))

    @property
    def bus(self) -> EventBus:
        """The process-wide event bus."""
        return self._bus

    def build_brain(self) -> Brain:
        """Assemble the brain from the configured LLM backend."""
        settings = self._settings
        if settings.anthropic_api_key is None:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env "
                "and add your key."
            )
        backend = AnthropicBackend(
            api_key=settings.anthropic_api_key.get_secret_value(),
            model=settings.llm.model,
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
        )
        history = ConversationBuffer(
            max_messages=settings.memory.short_term_max_messages
        )
        self._log.info("brain_ready", model=settings.llm.model)
        return Brain(
            llm=backend,
            history=history,
            system_prompt=build_system_prompt(settings.persona),
            bus=self._bus,
        )

    async def run_cli(self) -> None:
        """Run the text REPL until the user exits."""
        repl = CliRepl(self.build_brain(), assistant_name=self._settings.persona.name)
        try:
            await repl.run()
        finally:
            await self._bus.aclose()

    def _build_tts(self) -> tuple["TtsBackend", "TtsBackend | None"]:
        """Return (primary, fallback) TTS backends per config."""
        from jarvis.speech.tts import ElevenLabsTts, PiperTts, TtsBackend

        settings = self._settings
        tts_cfg = settings.speech.tts

        def piper() -> TtsBackend:
            return PiperTts(
                voices=tts_cfg.piper.voices,
                default_language=tts_cfg.piper.default_language,
            )

        if tts_cfg.backend == "piper":
            return piper(), None
        if settings.elevenlabs_api_key is None:
            raise ConfigError(
                "speech.tts.backend is 'elevenlabs' but ELEVENLABS_API_KEY "
                "is not set in .env."
            )
        primary: TtsBackend = ElevenLabsTts(
            api_key=settings.elevenlabs_api_key.get_secret_value(),
            voice_id=tts_cfg.elevenlabs.voice_id,
            model_id=tts_cfg.elevenlabs.model_id,
        )
        fallback = piper() if tts_cfg.fallback_to_local else None
        return primary, fallback

    def build_voice_pipeline(self) -> "VoicePipeline":
        """Assemble the full voice stack from configuration."""
        from jarvis.audio.capture import MicrophoneSource
        from jarvis.audio.endpointing import UtteranceCollector
        from jarvis.audio.playback import SoundDeviceSink
        from jarvis.audio.vad import EnergyVad, SileroVad, VoiceActivityDetector
        from jarvis.audio.wake import OpenWakeWordDetector
        from jarvis.interfaces.voice import VoicePipeline
        from jarvis.speech.stt import FasterWhisperStt

        settings = self._settings
        audio = settings.audio

        vad: VoiceActivityDetector
        if audio.vad.backend == "silero":
            vad = SileroVad()
        else:
            vad = EnergyVad(threshold=audio.vad.energy_threshold)

        tts, tts_fallback = self._build_tts()
        pipeline = VoicePipeline(
            source=MicrophoneSource(
                sample_rate=audio.sample_rate,
                frame_ms=audio.frame_ms,
                device=audio.input_device,
            ),
            wake=OpenWakeWordDetector(
                model=audio.wake.model,
                threshold=audio.wake.threshold,
                refractory_s=audio.wake.refractory_s,
            ),
            collector=UtteranceCollector(
                vad,
                speech_threshold=audio.vad.speech_threshold,
                silence_ms=audio.vad.silence_ms,
                no_speech_timeout_ms=audio.vad.no_speech_timeout_ms,
                max_utterance_ms=audio.vad.max_utterance_ms,
                min_speech_ms=audio.vad.min_speech_ms,
            ),
            stt=FasterWhisperStt(
                model_size=settings.speech.stt.model_size,
                device=settings.speech.stt.device,
                compute_type=settings.speech.stt.compute_type,
                allowed_languages=settings.persona.languages,
            ),
            tts=tts,
            tts_fallback=tts_fallback,
            sink=SoundDeviceSink(device=audio.output_device),
            brain=self.build_brain(),
            bus=self._bus,
        )
        self._log.info("voice_pipeline_built")
        return pipeline

    async def run_voice(self) -> None:
        """Run the wake-word -> STT -> brain -> TTS loop until cancelled."""
        pipeline = self.build_voice_pipeline()
        try:
            await pipeline.run()
        finally:
            await self._bus.aclose()
