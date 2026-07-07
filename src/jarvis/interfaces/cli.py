"""Interactive text REPL sharing the same brain as the voice pipeline."""

from __future__ import annotations

import asyncio
import sys

import structlog

from jarvis.brain.orchestrator import Brain, BrainError
from jarvis.events import new_trace_id
from jarvis.log import bind_trace_id

__all__ = ["CliRepl"]

_HELP = """\
Commands:
  /reset   forget the conversation and start over
  /help    show this help
  /quit    exit (Ctrl-D works too)\
"""


class CliRepl:
    """A minimal line-based chat loop.

    ``input()`` blocks, so it runs in a worker thread via
    ``asyncio.to_thread`` and never stalls the event loop.
    """

    def __init__(self, brain: Brain, *, assistant_name: str = "Jarvis") -> None:
        self._brain = brain
        self._name = assistant_name
        self._log = structlog.get_logger("jarvis.cli")

    async def run(self) -> None:
        """Run the REPL until /quit, Ctrl-D, or Ctrl-C."""
        bind_trace_id(new_trace_id())
        print(f"{self._name} ready. Type /help for commands.")
        while True:
            try:
                line = await asyncio.to_thread(input, "you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            line = line.strip()
            if not line:
                continue
            if line in {"/quit", "/exit"}:
                break
            if line == "/help":
                print(_HELP)
                continue
            if line == "/reset":
                self._brain.reset()
                bind_trace_id(new_trace_id())
                print("(conversation cleared)")
                continue

            await self._answer(line)
        print("Goodbye.")

    async def _answer(self, line: str) -> None:
        prefix = f"{self._name.lower()}> "
        sys.stdout.write(prefix)
        sys.stdout.flush()
        wrote_any = False
        try:
            async for chunk in self._brain.respond(line):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                wrote_any = True
        except BrainError as exc:
            if wrote_any:
                sys.stdout.write("\n")
            print(f"[error] {exc}")
            return
        sys.stdout.write("\n")
        sys.stdout.flush()
