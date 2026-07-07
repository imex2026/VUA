"""Console entry point: the ``jarvis`` command."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Sequence

from jarvis import __version__
from jarvis.app import JarvisApp
from jarvis.audio.types import AudioError
from jarvis.config import ConfigError
from jarvis.speech.stt import SttError
from jarvis.speech.tts import TtsError

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="Jarvis - a modular, async, voice-first AI assistant.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="chat",
        choices=["chat", "listen", "serve"],
        help=(
            "chat: text REPL (default). "
            "listen: voice pipeline (requires the [audio] extra). "
            "serve: HTTP API (Phase 5)."
        ),
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument("--version", action="version", version=f"jarvis {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the selected interface."""
    args = _build_parser().parse_args(argv)

    if args.command == "serve":
        print("'serve' is not available yet (it arrives in Phase 5).")
        return 2

    try:
        app = JarvisApp.from_config(args.config)
        if args.command == "listen":
            asyncio.run(app.run_voice())
        else:
            asyncio.run(app.run_cli())
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except (AudioError, SttError, TtsError) as exc:
        print(f"Voice stack error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
