"""Speech layer: STT (faster-whisper) and TTS (Piper, cloud opt-in)."""

from jarvis.speech.stt import FasterWhisperStt, SttBackend, SttError, Transcript
from jarvis.speech.tts import ElevenLabsTts, PiperTts, TtsBackend, TtsError

__all__ = [
    "ElevenLabsTts",
    "FasterWhisperStt",
    "PiperTts",
    "SttBackend",
    "SttError",
    "Transcript",
    "TtsBackend",
    "TtsError",
]
