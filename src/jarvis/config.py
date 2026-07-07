"""Configuration loading and validation.

All tunables live in a single ``config.yaml``; secrets come from the
environment / ``.env``. Precedence (highest first): environment
variables, ``.env``, ``config.yaml``, coded defaults. Nested values are
addressed in the environment with ``JARVIS_`` and ``__`` as delimiter,
e.g. ``JARVIS_LLM__MODEL=claude-opus-4-8``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

__all__ = [
    "AppSection",
    "AudioSection",
    "ConfigError",
    "ElevenLabsSection",
    "JarvisSettings",
    "LLMSection",
    "MemorySection",
    "PersonaSection",
    "PiperSection",
    "SpeechSection",
    "SttSection",
    "TtsSection",
    "VadSection",
    "WakeSection",
    "load_settings",
]


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


class AppSection(BaseModel):
    """Process-wide settings."""

    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"


class LLMSection(BaseModel):
    """Which model powers the brain and how it is sampled."""

    backend: Literal["anthropic"] = "anthropic"
    model: str = "claude-sonnet-5"
    max_tokens: int = Field(default=1024, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)


class PersonaSection(BaseModel):
    """How Jarvis presents itself."""

    name: str = "Jarvis"
    languages: list[str] = Field(default_factory=lambda: ["en", "fr", "de", "ar"])
    extra_instructions: str = ""


class MemorySection(BaseModel):
    """Conversation memory sizing (long-term store arrives in Phase 4)."""

    short_term_max_messages: int = Field(default=80, gt=1)


class WakeSection(BaseModel):
    """Wake-word detection."""

    backend: Literal["openwakeword"] = "openwakeword"
    model: str = "hey_jarvis"
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    refractory_s: float = Field(default=2.0, ge=0.0)


class VadSection(BaseModel):
    """Voice activity detection and utterance endpointing."""

    backend: Literal["silero", "energy"] = "silero"
    speech_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    energy_threshold: float = Field(default=0.015, gt=0.0)
    silence_ms: float = Field(default=800.0, gt=0.0)
    no_speech_timeout_ms: float = Field(default=6000.0, gt=0.0)
    max_utterance_ms: float = Field(default=30000.0, gt=0.0)
    min_speech_ms: float = Field(default=300.0, ge=0.0)


class AudioSection(BaseModel):
    """Microphone and speaker devices plus the audio sub-layers."""

    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = Field(default=16000, gt=0)
    frame_ms: int = Field(default=80, gt=0)
    wake: WakeSection = Field(default_factory=WakeSection)
    vad: VadSection = Field(default_factory=VadSection)


class SttSection(BaseModel):
    """Speech-to-text backend selection and model sizing."""

    backend: Literal["faster_whisper"] = "faster_whisper"
    model_size: str = "small"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: str = "auto"


class PiperSection(BaseModel):
    """Local Piper voices, one .onnx file per language."""

    voices: dict[str, str] = Field(default_factory=dict)
    default_language: str = "en"


class ElevenLabsSection(BaseModel):
    """Opt-in cloud TTS; the API key lives in .env."""

    voice_id: str = ""
    model_id: str = "eleven_multilingual_v2"


class TtsSection(BaseModel):
    """Text-to-speech backend selection.

    When a cloud backend is primary and ``fallback_to_local`` is true,
    Piper takes over on failure and the switch is announced aloud.
    """

    backend: Literal["piper", "elevenlabs"] = "piper"
    fallback_to_local: bool = True
    piper: PiperSection = Field(default_factory=PiperSection)
    elevenlabs: ElevenLabsSection = Field(default_factory=ElevenLabsSection)


class SpeechSection(BaseModel):
    """STT + TTS configuration."""

    stt: SttSection = Field(default_factory=SttSection)
    tts: TtsSection = Field(default_factory=TtsSection)


class JarvisSettings(BaseSettings):
    """Root settings object assembled from yaml + environment."""

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSection = Field(default_factory=AppSection)
    llm: LLMSection = Field(default_factory=LLMSection)
    persona: PersonaSection = Field(default_factory=PersonaSection)
    memory: MemorySection = Field(default_factory=MemorySection)
    audio: AudioSection = Field(default_factory=AudioSection)
    speech: SpeechSection = Field(default_factory=SpeechSection)

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "JARVIS_ANTHROPIC_API_KEY"),
    )
    elevenlabs_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "ELEVENLABS_API_KEY", "JARVIS_ELEVENLABS_API_KEY"
        ),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Let the environment override values passed from config.yaml.

        ``load_settings`` feeds the yaml contents through ``init``
        kwargs, which pydantic-settings would normally rank highest;
        reordering puts env and .env above the file.
        """
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)


def load_settings(config_path: str | Path | None = None) -> JarvisSettings:
    """Build :class:`JarvisSettings` from ``config.yaml`` + environment.

    A missing file is fine (defaults apply); a file that is not a
    mapping raises :class:`ConfigError`.
    """
    path = Path(config_path) if config_path is not None else Path("config.yaml")
    data: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path} must contain a YAML mapping at the top level")
        data = loaded
    try:
        return JarvisSettings(**data)
    except ValueError as exc:
        raise ConfigError(f"Invalid configuration in {path}: {exc}") from exc
