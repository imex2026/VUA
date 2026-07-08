"""The voice interface: wake word -> record -> transcribe -> think -> speak.

One asyncio task drives the pipeline; model inference happens in worker
threads inside the backends, so the loop stays free. Stage transitions
are published on the EventBus (WakeDetected, UtteranceCaptured,
TranscriptReady; the brain itself publishes SpeechOutput) for
observability and extension points, while the data flows directly
through the pipeline so the turn-taking state machine stays
deterministic.
"""

from __future__ import annotations

import time

import structlog

from jarvis.audio.capture import AudioSource
from jarvis.audio.endpointing import NoSpeech, Utterance, UtteranceCollector
from jarvis.audio.playback import AudioSink
from jarvis.audio.types import AudioClip
from jarvis.audio.wake import WakeWordDetector
from jarvis.brain.orchestrator import Brain, BrainError
from jarvis.events import (
    EventBus,
    TranscriptReady,
    UtteranceCaptured,
    WakeDetected,
    new_trace_id,
)
from jarvis.log import bind_trace_id
from jarvis.speech.stt import SttBackend, SttError
from jarvis.speech.tts import TtsBackend, TtsError

__all__ = ["VoicePipeline"]

_DEGRADATION_NOTICE = (
    "My cloud voice is unreachable, so I'm switching to the local one."
)


class VoicePipeline:
    """Turn-taking loop between the microphone and the brain.

    While Jarvis is transcribing, thinking, or speaking, microphone
    frames are discarded (no barge-in yet); frames captured before
    speaking finished are dropped by timestamp so Jarvis never wakes
    itself with its own voice.
    """

    def __init__(
        self,
        *,
        source: AudioSource,
        wake: WakeWordDetector,
        collector: UtteranceCollector,
        stt: SttBackend,
        tts: TtsBackend,
        sink: AudioSink,
        brain: Brain,
        bus: EventBus,
        tts_fallback: TtsBackend | None = None,
    ) -> None:
        self._source = source
        self._wake = wake
        self._collector = collector
        self._stt = stt
        self._tts = tts
        self._tts_fallback = tts_fallback
        self._sink = sink
        self._brain = brain
        self._bus = bus
        self._collecting = False
        self._ignore_before = 0.0
        self._degradation_announced = False
        self._log = structlog.get_logger("jarvis.voice")

    async def run(self) -> None:
        """Consume the audio source until it ends or is cancelled."""
        self._log.info("voice_pipeline_ready")
        async for frame in self._source.frames():
            if frame.captured_at < self._ignore_before:
                continue  # recorded while we were busy or speaking
            if not self._collecting:
                detection = await self._wake.process(frame)
                if detection is not None:
                    bind_trace_id(new_trace_id())
                    self._log.info(
                        "wake_detected",
                        wake_word=detection.wake_word,
                        score=round(detection.score, 3),
                    )
                    self._bus.publish(
                        WakeDetected(
                            wake_word=detection.wake_word,
                            score=detection.score,
                        )
                    )
                    self._collector.reset()
                    self._collecting = True
                continue

            result = await self._collector.feed(frame)
            if result is None:
                continue
            self._collecting = False
            if isinstance(result, NoSpeech):
                self._log.info("no_speech", reason=result.reason)
            else:
                await self._handle_utterance(result)
            self._wake.reset()
            self._ignore_before = time.monotonic()

    async def _handle_utterance(self, utterance: Utterance) -> None:
        clip = AudioClip(pcm=utterance.pcm, sample_rate=utterance.sample_rate)
        self._bus.publish(
            UtteranceCaptured(audio=clip.pcm, sample_rate=clip.sample_rate)
        )
        try:
            transcript = await self._stt.transcribe(clip)
        except SttError as exc:
            self._log.error("stt_failed", error=str(exc))
            return
        if not transcript.text.strip():
            self._log.info("empty_transcript")
            return
        self._bus.publish(
            TranscriptReady(text=transcript.text, language=transcript.language)
        )

        parts: list[str] = []
        try:
            async for chunk in self._brain.respond(transcript.text):
                parts.append(chunk)
        except BrainError as exc:
            await self._speak(str(exc), transcript.language)
            return
        reply = "".join(parts)
        if reply:
            await self._speak(reply, transcript.language)

    async def _speak(self, text: str, language: str) -> None:
        """Synthesize and play, degrading to the fallback backend aloud."""
        try:
            clip = await self._tts.synthesize(text, language)
        except TtsError as exc:
            self._log.warning("tts_failed", backend=self._tts.name, error=str(exc))
            if self._tts_fallback is None:
                self._log.error("tts_no_fallback")
                return
            try:
                if not self._degradation_announced:
                    self._degradation_announced = True
                    notice = await self._tts_fallback.synthesize(
                        _DEGRADATION_NOTICE, "en"
                    )
                    await self._sink.play(notice)
                clip = await self._tts_fallback.synthesize(text, language)
            except TtsError as fallback_exc:
                self._log.error(
                    "tts_fallback_failed",
                    backend=self._tts_fallback.name,
                    error=str(fallback_exc),
                )
                return
        await self._sink.play(clip)
