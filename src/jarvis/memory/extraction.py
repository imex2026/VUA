"""Automatic fact extraction from conversation turns.

After each exchange, a cheap LLM call decides whether anything durable
was said (preferences, personal details, projects, commitments) and
distills it into short standalone facts for the long-term store.
"""

from __future__ import annotations

import structlog

from jarvis.brain.llm import ChatMessage, LLMBackend, TextDelta

__all__ = ["FactExtractor"]

_SYSTEM = """\
You maintain the long-term memory of a personal voice assistant.
Given one conversation turn, extract durable facts about the user that
will still matter in future conversations: preferences, personal
details, people, ongoing projects, habits, commitments.

Rules:
- Only genuinely durable facts. No transient chitchat, no weather, no
  one-off questions, nothing the assistant said about itself.
- Each fact is one short, standalone sentence naming the user, e.g.
  "The user's daughter is called Lina."
- At most {max_facts} facts.

Output one fact per line with no bullets or numbering.
If nothing is worth remembering, output exactly: NONE\
"""


class FactExtractor:
    """Distills a conversation turn into memorable facts via the LLM."""

    def __init__(self, llm: LLMBackend, *, max_facts: int = 5) -> None:
        self._llm = llm
        self._max_facts = max_facts
        self._log = structlog.get_logger("jarvis.memory.extract")

    async def extract(self, user_text: str, assistant_text: str) -> list[str]:
        """Return facts worth remembering from this turn (may be [])."""
        prompt = f"User said: {user_text}\n\n" f"Assistant replied: {assistant_text}"
        parts: list[str] = []
        async for event in self._llm.stream(
            system=_SYSTEM.format(max_facts=self._max_facts),
            messages=[ChatMessage(role="user", content=prompt)],
        ):
            if isinstance(event, TextDelta):
                parts.append(event.text)
        facts = self.parse("".join(parts), self._max_facts)
        self._log.info("facts_extracted", count=len(facts))
        return facts

    @staticmethod
    def parse(output: str, max_facts: int) -> list[str]:
        """Parse the model's line-based output into clean facts."""
        facts: list[str] = []
        for raw_line in output.splitlines():
            line = raw_line.strip().lstrip("-*•").strip()
            if not line or line.upper() == "NONE":
                continue
            facts.append(line)
            if len(facts) >= max_facts:
                break
        return facts
