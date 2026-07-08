"""Audio layer: capture, wake word, VAD, endpointing, playback.

Heavy dependencies (sounddevice, openwakeword, silero-vad) are imported
lazily inside the backends, so this package is importable without the
``audio`` extra - useful for tests and text-only deployments.
"""

from jarvis.audio.capture import AudioSource, MicrophoneSource
from jarvis.audio.endpointing import NoSpeech, Utterance, UtteranceCollector
from jarvis.audio.playback import AudioSink, SoundDeviceSink
from jarvis.audio.types import AudioClip, AudioError, AudioFrame
from jarvis.audio.vad import EnergyVad, SileroVad, VoiceActivityDetector
from jarvis.audio.wake import OpenWakeWordDetector, WakeDetection, WakeWordDetector

__all__ = [
    "AudioClip",
    "AudioError",
    "AudioFrame",
    "AudioSink",
    "AudioSource",
    "EnergyVad",
    "MicrophoneSource",
    "NoSpeech",
    "OpenWakeWordDetector",
    "SileroVad",
    "SoundDeviceSink",
    "Utterance",
    "UtteranceCollector",
    "VoiceActivityDetector",
    "WakeDetection",
    "WakeWordDetector",
]
