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
    "ConfigError",
    "JarvisSettings",
    "LLMSection",
    "MemorySection",
    "PersonaSection",
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

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "JARVIS_ANTHROPIC_API_KEY"),
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
