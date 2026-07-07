"""Composition root: wires configuration, logging, bus, brain, interfaces."""

from __future__ import annotations

from pathlib import Path

import structlog

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
