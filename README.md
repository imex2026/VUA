# Jarvis

A modular, async, voice-first AI assistant. Wake word → speech-to-text →
Claude with tools and memory → text-to-speech, all on one asyncio event
loop, with a CLI REPL and HTTP API sharing the same brain.

Local/offline by default (openWakeWord, Silero VAD, faster-whisper,
Piper); cloud services (Claude, ElevenLabs, search APIs) are opt-in per
config, with graceful fallback to local backends when a cloud service is
unreachable.

## Architecture

```mermaid
flowchart TB
    subgraph Interfaces["Interface layer"]
        VOICE[Voice pipeline]
        CLI[CLI REPL]
        API[FastAPI /chat]
    end

    subgraph Audio["Audio layer (Phase 2)"]
        MIC[Mic capture<br/>sounddevice] --> WAKE[Wake word<br/>openWakeWord]
        MIC --> VAD[Silero VAD]
    end

    subgraph Speech["Speech layer (Phase 2)"]
        STT[STT protocol<br/>faster-whisper, auto-lang ar/fr/en/de]
        TTS[TTS protocol<br/>Piper local - cloud opt-in]
    end

    BUS{{EventBus<br/>WakeDetected - TranscriptReady -<br/>ToolCallRequested - SpeechOutput}}

    subgraph Brain["Brain layer"]
        ORCH[Orchestrator<br/>streaming + tool-use loop]
        LLM[LLM protocol<br/>Anthropic SDK]
    end

    subgraph Tools["Tool layer (Phase 3)"]
        REG[Tool registry]
        REG --> BUILTIN[web search - calendar -<br/>smart home - files]
        REG --> MCP[MCP client adapter]
    end

    subgraph Memory["Memory layer"]
        STM[Short-term buffer]
        LTM[Long-term RAG (Phase 4)<br/>sentence-transformers + Chroma]
    end

    WAKE -->|WakeDetected| BUS
    VAD -->|utterance| STT
    STT -->|TranscriptReady| BUS
    BUS --> ORCH
    VOICE --> BUS
    CLI --> ORCH
    API --> ORCH
    ORCH <--> LLM
    ORCH <--> REG
    ORCH <--> STM
    ORCH <--> LTM
    ORCH -->|SpeechOutput| BUS
    BUS --> TTS
```

Design rules:

- **Single asyncio loop.** Nothing blocks the hot path; CPU-bound work
  (STT/TTS inference, embeddings) runs in executors, `input()` in a
  worker thread.
- **Typed events on an EventBus** decouple the layers
  (`src/jarvis/events.py`). Every event carries a `trace_id` that also
  tags every log line for that conversation.
- **Every external dependency behind a `Protocol`** (`LLMBackend`,
  later `SttBackend`, `TtsBackend`, `VectorStore`) so backends swap via
  `config.yaml`.
- **One config file** (`config.yaml`, validated by pydantic-settings);
  secrets only in `.env`; environment variables override the file
  (`JARVIS_LLM__MODEL=...`).

## Project status

| Phase | Scope | Status |
|---|---|---|
| 1 | Skeleton, config, logging, EventBus, Claude brain, CLI REPL | done |
| 2 | Audio (wake word, VAD) + speech (faster-whisper STT, Piper TTS) | done |
| 3 | Tool registry, built-in tools, MCP client adapter | planned |
| 4 | Long-term semantic memory (RAG) with fact extraction | planned |
| 5 | FastAPI interface, mypy strict pass, full test suite, docs | planned |

## Setup

Requires Python 3.11+. Works on Linux and Windows.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp .env.example .env               # then put your ANTHROPIC_API_KEY in it
jarvis chat
```

In the REPL: `/reset` clears the conversation, `/help` lists commands,
`/quit` (or Ctrl-D) exits. `jarvis --config path/to/config.yaml chat`
uses an alternate config.

### Voice mode

```bash
pip install -e ".[audio]"          # sounddevice, openwakeword, silero-vad,
                                   # faster-whisper, piper-tts
```

Then download at least one Piper voice from
<https://huggingface.co/rhasspy/piper-voices> (both the `.onnx` and its
`.onnx.json` sidecar), point `speech.tts.piper.voices` in `config.yaml`
at it, and run:

```bash
jarvis listen
```

Say **"hey jarvis"**, wait for the log line, and speak. Notes:

- The openWakeWord and whisper models download automatically on first
  run; the first `listen` needs network once, afterwards everything is
  offline.
- STT auto-detects Arabic, French, English, and German, and the reply
  is synthesized with the Piper voice mapped to that language (falling
  back to `default_language`).
- With a CUDA GPU, `speech.stt.device: auto` picks it up automatically;
  bump `model_size` to `medium` or `large-v3` for better accuracy.
- To use ElevenLabs for speech output, set `speech.tts.backend:
  elevenlabs`, a `voice_id`, and `ELEVENLABS_API_KEY` in `.env`. If the
  cloud is unreachable, Jarvis switches to Piper and says so aloud
  (`fallback_to_local: true`).
- No barge-in yet: the mic is ignored while Jarvis is thinking or
  speaking (frames are dropped by timestamp so it can't wake itself).

## Development

```bash
black src tests
flake8 src tests
mypy
pytest
```

The test suite mocks the LLM (and later the audio stack) — it needs no
API key and no network.

## Layout

```
src/jarvis/
├── __main__.py      # `jarvis` console script
├── app.py           # composition root
├── config.py        # pydantic-settings models, yaml + env loading
├── events.py        # typed events + async EventBus
├── log.py           # structlog setup, trace IDs
├── brain/           # llm.py (protocol + Anthropic), orchestrator.py, persona.py
├── memory/          # short_term.py (long-term RAG arrives in Phase 4)
├── interfaces/      # cli.py, voice.py (HTTP in Phase 5)
├── audio/           # capture, wake word, VAD, endpointing, playback
├── speech/          # stt.py (faster-whisper), tts.py (Piper/ElevenLabs)
└── tools/           # Phase 3
```
