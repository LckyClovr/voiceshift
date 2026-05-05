import asyncio
import io
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, load_silero_vad

import edge_tts
from elevenlabs.client import ElevenLabs


# Native sample rate of the ElevenLabs PCM output we request.
# Keeping it at 22050 minimizes bandwidth without audible quality loss for speech.
ELEVEN_PCM_SAMPLE_RATE = 22050
ELEVEN_PCM_FORMAT = "pcm_22050"


@dataclass
class Voice:
    name: str
    voice_id: str


@dataclass
class Config:
    sample_rate: int = 16000
    block_duration: float = 0.25
    rolling_buffer_seconds: int = 4
    vad_threshold: float = 0.4
    min_speech_duration: float = 0.2
    buffer_before_speech: float = 0.3
    silence_end_ms: int = 350
    whisper_model: str = "tiny.en"
    edge_voice: str = "en-US-GuyNeural"
    use_elevenlabs: bool = True
    eleven_voice_id: str = ""
    eleven_model_id: str = "eleven_flash_v2_5"
    eleven_api_key: str = ""
    voices: list[Voice] = field(default_factory=list)


def parse_voices(raw: str) -> list[Voice]:
    voices: list[Voice] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        name, voice_id = entry.split(":", 1)
        name, voice_id = name.strip(), voice_id.strip()
        if name and voice_id:
            voices.append(Voice(name=name, voice_id=voice_id))
    return voices


def load_config() -> Config:
    load_dotenv()
    return Config(
        whisper_model=os.environ.get("WHISPER_MODEL", "tiny.en"),
        edge_voice=os.environ.get("EDGE_VOICE", "en-US-GuyNeural"),
        use_elevenlabs=os.environ.get("USE_ELEVENLABS", "true").lower() == "true",
        eleven_voice_id=os.environ.get("ELEVEN_VOICE_ID", ""),
        eleven_model_id=os.environ.get("ELEVEN_MODEL_ID", "eleven_flash_v2_5"),
        eleven_api_key=os.environ.get("ELEVEN_API_KEY", ""),
        voices=parse_voices(os.environ.get("ELEVEN_VOICES", "")),
        silence_end_ms=int(os.environ.get("SILENCE_END_MS", "350")),
    )


def pick_voice(config: Config) -> str:
    if not config.voices:
        if not config.eleven_voice_id:
            raise RuntimeError(
                "No voices configured. Set ELEVEN_VOICES (recommended) or "
                "ELEVEN_VOICE_ID in your .env."
            )
        return config.eleven_voice_id

    print("\nAvailable voices:")
    for i, voice in enumerate(config.voices):
        print(f"  {i}: {voice.name}")
    while True:
        raw = input(
            f"Choose voice index (blank for 0 — {config.voices[0].name}): "
        ).strip()
        if not raw:
            return config.voices[0].voice_id
        try:
            idx = int(raw)
            if 0 <= idx < len(config.voices):
                return config.voices[idx].voice_id
        except ValueError:
            pass
        print(f"Enter a number 0–{len(config.voices) - 1}.")


def detect_device() -> tuple[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


class VoiceTransformer:
    def __init__(
        self,
        config: Config,
        voice_id: str,
        output_device_idx: int | None = None,
    ):
        self.config = config
        self.voice_id = voice_id
        self.output_device_idx = output_device_idx

        device, compute_type = detect_device()
        print(f"Loading Whisper ({config.whisper_model}) on {device}...")
        self.whisper = WhisperModel(
            config.whisper_model, device=device, compute_type=compute_type
        )
        self.vad_model = load_silero_vad()

        self.eleven_client: ElevenLabs | None = None
        if config.use_elevenlabs:
            if not config.eleven_api_key:
                raise RuntimeError(
                    "USE_ELEVENLABS=true but ELEVEN_API_KEY is not set. "
                    "Add it to your .env file."
                )
            self.eleven_client = ElevenLabs(api_key=config.eleven_api_key)

        self.audio_queue: queue.Queue = queue.Queue()
        self.vad_buffer: list[np.ndarray] = []
        self.silence_buffer: list[np.ndarray] = []

        self.in_speech = False
        self.processing_lock = asyncio.Lock()

        # Suppresses mic input while TTS is playing so the synthesized voice
        # doesn't get fed back through the mic and re-transcribed.
        self.muting_until = 0.0

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        if time.time() < self.muting_until:
            return
        self.audio_queue.put(indata.copy())

    def stream_pcm_playback(self, stream, pcm_chunks, t_request_sent: float) -> None:
        """Pump PCM chunks from the network into a pre-opened output stream.

        The output stream is opened *before* the network request goes out so
        the device is already hot when the first chunk lands.
        """
        first_chunk_logged = False
        try:
            for chunk in pcm_chunks:
                if not chunk:
                    continue
                if not first_chunk_logged:
                    ttfa_ms = (time.time() - t_request_sent) * 1000
                    print(f"  Time-to-first-audio: {ttfa_ms:.0f}ms")
                    first_chunk_logged = True
                stream.write(chunk)
        except Exception as e:
            print(f"Playback error: {e}")
            traceback.print_exc()

    async def synthesize_and_play_elevenlabs(self, text: str) -> None:
        assert self.eleven_client is not None
        # Mute the mic for the full TTS lifecycle. We extend muting_until
        # while playback runs and add a small grace period after, so the
        # tail of the synthesized voice can't bleed into the mic and
        # trigger a feedback loop.
        self.muting_until = time.time() + 60  # extended each chunk below

        # Pre-open the output device so playback starts the instant the
        # first network chunk arrives — opening lazily on first write()
        # would add 50–150ms on Windows.
        out_stream = sd.RawOutputStream(
            samplerate=ELEVEN_PCM_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=self.output_device_idx,
        )
        out_stream.start()

        try:
            t_request = time.time()
            pcm_iter = self.eleven_client.text_to_speech.convert_as_stream(
                text=text,
                voice_id=self.voice_id,
                model_id=self.config.eleven_model_id,
                output_format=ELEVEN_PCM_FORMAT,
            )
            # Run the blocking iterator in a worker thread so the event loop
            # stays responsive to incoming microphone audio.
            await asyncio.to_thread(
                self.stream_pcm_playback, out_stream, pcm_iter, t_request
            )
        finally:
            out_stream.stop()
            out_stream.close()
            # Grace period after playback ends: drop any mic frames that
            # are still echoing the tail of the synthesized audio, then
            # also discard anything queued during muting so the next loop
            # iteration starts clean.
            self.muting_until = time.time() + 0.3
            await asyncio.sleep(0.3)
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            self.vad_buffer = []
            self.silence_buffer = []

    async def synthesize_and_play_edge(self, text: str) -> None:
        tts = edge_tts.Communicate(text, self.config.edge_voice)
        buf = io.BytesIO()
        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        data, fs = sf.read(buf, dtype="float32")
        await asyncio.to_thread(self._play_decoded, data, fs)

    def _play_decoded(self, data: np.ndarray, fs: int) -> None:
        sd.play(data, fs, device=self.output_device_idx)
        sd.wait()

    async def transcribe_and_speak(self, audio_np: np.ndarray) -> None:
        async with self.processing_lock:
            try:
                t0 = time.time()
                segments, _ = self.whisper.transcribe(audio_np, beam_size=1)
                text = " ".join(seg.text.strip() for seg in segments).strip()
                t_stt = time.time() - t0
                if not text:
                    return

                print(f"Transcribed ({t_stt * 1000:.0f}ms): {text}")

                t1 = time.time()
                if self.config.use_elevenlabs:
                    await self.synthesize_and_play_elevenlabs(text)
                else:
                    await self.synthesize_and_play_edge(text)
                t_tts = time.time() - t1
                print(f"  TTS+playback: {t_tts * 1000:.0f}ms")
            except Exception:
                print("Error in transcription/synthesis:")
                traceback.print_exc()

    async def process_audio_loop(self) -> None:
        cfg = self.config
        silence_buffer_size = int(cfg.buffer_before_speech / cfg.block_duration)
        rolling_buffer_blocks = int(cfg.rolling_buffer_seconds / cfg.block_duration)
        silence_end_seconds = cfg.silence_end_ms / 1000.0

        print("Listening...")
        while True:
            try:
                if self.audio_queue.empty():
                    await asyncio.sleep(0.01)
                    continue

                chunk = self.audio_queue.get()
                audio_mono = chunk[:, 0] if chunk.ndim > 1 else chunk

                self.silence_buffer.append(audio_mono)
                if len(self.silence_buffer) > silence_buffer_size:
                    self.silence_buffer.pop(0)

                self.vad_buffer.append(audio_mono)
                if len(self.vad_buffer) > rolling_buffer_blocks:
                    self.vad_buffer.pop(0)

                if self.processing_lock.locked():
                    continue

                # VAD needs a multi-second window to classify reliably, so we
                # always run it on the full rolling buffer and use the position
                # of the last detected speech segment to decide if speech is
                # still happening or has ended.
                audio_concat = np.concatenate(self.vad_buffer)
                speech_segments = get_speech_timestamps(
                    audio_concat,
                    self.vad_model,
                    sampling_rate=cfg.sample_rate,
                    threshold=cfg.vad_threshold,
                    return_seconds=True,
                    min_speech_duration_ms=int(cfg.min_speech_duration * 1000),
                )

                if not speech_segments:
                    self.in_speech = False
                    continue

                self.in_speech = True
                last_segment = speech_segments[-1]
                buffer_duration = len(audio_concat) / cfg.sample_rate
                trailing_silence = buffer_duration - last_segment["end"]

                if trailing_silence < silence_end_seconds:
                    continue

                # Endpoint: enough trailing silence after the last speech segment.
                self.in_speech = False
                segment_audio = self._extract_speech_audio(
                    audio_concat, speech_segments
                )
                asyncio.create_task(self.transcribe_and_speak(segment_audio))
                self.vad_buffer = []

            except Exception:
                print("Error in process_audio_loop:")
                traceback.print_exc()
                await asyncio.sleep(0.1)

    def _extract_speech_audio(
        self,
        audio_concat: np.ndarray,
        speech_segments: list,
    ) -> np.ndarray:
        """Cut from the start of the first segment (with pre-roll) to the end
        of the last segment, so we don't send trailing silence to Whisper."""
        cfg = self.config
        first_start = speech_segments[0]["start"]
        last_end = speech_segments[-1]["end"]

        start_sample = max(
            0,
            int(first_start * cfg.sample_rate)
            - int(cfg.buffer_before_speech * cfg.sample_rate),
        )
        end_sample = min(int(last_end * cfg.sample_rate), len(audio_concat))

        if start_sample > 0:
            return audio_concat[start_sample:end_sample]

        # Pre-roll spills into the silence_buffer (audio that's older than
        # the rolling VAD buffer) — pull from there to avoid clipping the
        # first phoneme.
        silence_audio = (
            np.concatenate(self.silence_buffer)
            if self.silence_buffer
            else np.zeros(0, dtype=np.float32)
        )
        silence_needed = min(
            len(silence_audio),
            int(cfg.buffer_before_speech * cfg.sample_rate),
        )
        return np.concatenate(
            [silence_audio[-silence_needed:], audio_concat[:end_sample]]
        )

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.create_task(self.process_audio_loop())

        def stream_thread():
            try:
                with sd.InputStream(
                    callback=self.audio_callback,
                    channels=1,
                    samplerate=self.config.sample_rate,
                    blocksize=int(self.config.sample_rate * self.config.block_duration),
                ):
                    print("Audio stream started. Speak now (Ctrl+C to quit).")
                    while True:
                        time.sleep(0.1)
            except Exception:
                print("Stream thread crashed:")
                traceback.print_exc()

        threading.Thread(target=stream_thread, daemon=True).start()
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


def list_output_devices() -> list[tuple[int, str]]:
    return [
        (i, d["name"])
        for i, d in enumerate(sd.query_devices())
        if d["max_output_channels"] > 0
    ]


def main() -> None:
    config = load_config()

    print("Available output devices:")
    for i, name in list_output_devices():
        print(f"  {i}: {name}")
    raw = input("Choose output device index (blank for system default): ").strip()
    output_device_idx = int(raw) if raw else None

    voice_id = pick_voice(config) if config.use_elevenlabs else ""

    transformer = VoiceTransformer(
        config, voice_id=voice_id, output_device_idx=output_device_idx
    )
    transformer.run()


if __name__ == "__main__":
    main()
