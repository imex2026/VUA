"""Unit tests for configuration loading and precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import ConfigError, load_settings


def test_defaults_without_config_file(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "missing.yaml")

    assert settings.llm.backend == "anthropic"
    assert settings.llm.model == "claude-sonnet-5"
    assert settings.persona.name == "Jarvis"


def test_yaml_values_are_applied(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "llm:\n  model: claude-opus-4-8\n  temperature: 0.2\n"
        "persona:\n  name: Vendredi\n",
        encoding="utf-8",
    )

    settings = load_settings(config)

    assert settings.llm.model == "claude-opus-4-8"
    assert settings.llm.temperature == 0.2
    assert settings.persona.name == "Vendredi"


def test_environment_overrides_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("llm:\n  model: from-yaml\n", encoding="utf-8")
    monkeypatch.setenv("JARVIS_LLM__MODEL", "from-env")

    settings = load_settings(config)

    assert settings.llm.model == "from-env"


def test_api_key_comes_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")

    settings = load_settings(tmp_path / "missing.yaml")

    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-test-123"
    # And it never leaks via repr.
    assert "sk-test-123" not in repr(settings)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_settings(config)


def test_invalid_values_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("llm:\n  temperature: 9.5\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_settings(config)
