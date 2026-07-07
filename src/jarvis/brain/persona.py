"""Jarvis's system persona."""

from __future__ import annotations

from jarvis.config import PersonaSection

__all__ = ["build_system_prompt"]

_LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "ar": "Arabic",
}

_TEMPLATE = """\
You are {name}, a voice-first personal assistant.

Style:
- Be concise. Your answers are usually spoken aloud, so default to a \
few sentences; expand only when the user asks for depth.
- Be competent and direct. Lead with the answer, not preamble.
- A light touch of wit is welcome; never at the expense of clarity, \
and never sycophantic.

Language:
- Detect the user's language and reply in it. You are fluent in \
{languages}.
- If a message mixes languages, follow the dominant one.

Honesty:
- If you don't know or a capability is unavailable, say so plainly and \
suggest the closest thing you can do.\
"""


def build_system_prompt(persona: PersonaSection) -> str:
    """Render the system prompt from the persona configuration."""
    languages = ", ".join(_LANGUAGE_NAMES.get(code, code) for code in persona.languages)
    prompt = _TEMPLATE.format(name=persona.name, languages=languages)
    if persona.extra_instructions.strip():
        prompt += "\n\n" + persona.extra_instructions.strip()
    return prompt
