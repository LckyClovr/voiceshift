import asyncio
import io
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, load_silero_vad

import edge_tts
from elevenlabs.client import ElevenLabs


@dataclass
class Config:
    sample_rate: int = 16000
    block_duration: float = 0.25
    rolling_buffer_seconds: int = 4
    vad_threshold: float = 0.4
    min_speech_duration: float = 0.2
    buffer_before_speech: float = 0.75
    whisper_model: str = "tiny.en"
    edge_voice: str = "en-US-GuyNeural"
    use_elevenlabs: bool = True
    eleven_voice_id: str = ""
    eleven_model_id: str = "eleven_multilingual_v2"
    eleven_api_key: str = ""


def load_config() -> Config:
    load_dotenv()
    return Config(
        whisper_model=os.environ.get("WHISPER_MODEL", "tiny.en"),
        edge_voice=os.environ.get("EDGE_VOICE", "en-US-GuyNeural"),
        use_elevenlabs=os.environ.get("USE_ELEVENLABS", "true").lower() == "true",
        eleven_voice_id=os.environ.get("ELEVEN_VOICE_ID", ""),
        eleven_model_id=os.environ.get("ELEVEN_MODEL_ID", "eleven_multilingual_v2"),
        eleven_api_key=os.environ.get("ELEVEN_API_KEY", ""),
    )


def detect_device() -> tuple[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


class VoiceTransformer:
    def __init__(self, config: Config, output_device_idx: int | None = None):
        self.config = config
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
            if not config.eleven_voice_id:
                raise RuntimeError(
                    "USE_ELEVENLABS=true but ELEVEN_VOICE_ID is not set."
                )
            self.eleven_client = ElevenLabs(api_key=config.eleven_api_key)

        self.audio_queue: queue.Queue = queue.Queue()
        self.vad_buffer: list[np.ndarray] = []
        self.silence_buffer: list[np.ndarray] = []
        self.is_speech_active = False
        self.speech_start_time = 0.0
        self.processing_lock = asyncio.Lock()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())

    def play_audio_bytes(self, audio_bytes: bytes) -> None:
        try:
            data, fs = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            sd.play(data, fs, device=self.output_device_idx)
            sd.wait()
        except Exception as e:
            print(f"Playback error: {e}")
            traceback.print_exc()

    async def synthesize_elevenlabs(self, text: str) -> bytes:
        assert self.eleven_client is not None
        audio_generator = self.eleven_client.text_to_speech.convert(
            text=text,
            voice_id=self.config.eleven_voice_id,
            model_id=self.config.eleven_model_id,
        )
        return b"".join(audio_generator)

    async def synthesize_edge(self, text: str) -> bytes:
        tts = edge_tts.Communicate(text, self.config.edge_voice)
        buf = io.BytesIO()
        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    async def transcribe_and_speak(self, audio_np: np.ndarray) -> None:
        async with self.processing_lock:
            try:
                segments, _ = self.whisper.transcribe(audio_np, beam_size=1)
                text = " ".join(seg.text.strip() for seg in segments).strip()
                if not text:
                    return

                print(f"Transcribed: {text}")

                if self.config.use_elevenlabs:
                    audio_data = await self.synthesize_elevenlabs(text)
                else:
                    audio_data = await self.synthesize_edge(text)

                self.play_audio_bytes(audio_data)
            except Exception:
                print("Error in transcription/synthesis:")
                traceback.print_exc()

    async def process_audio_loop(self) -> None:
        cfg = self.config
        silence_buffer_size = int(cfg.buffer_before_speech / cfg.block_duration)
        rolling_buffer_blocks = int(cfg.rolling_buffer_seconds / cfg.block_duration)

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

                audio_concat = np.concatenate(self.vad_buffer)
                speech_segments = get_speech_timestamps(
                    audio_concat,
                    self.vad_model,
                    sampling_rate=cfg.sample_rate,
                    threshold=cfg.vad_threshold,
                    return_seconds=True,
                )

                if not speech_segments:
                    self.is_speech_active = False
                    continue

                seg = speech_segments[0]
                start_time, end_time = seg["start"], seg["end"]

                if not self.is_speech_active:
                    self.is_speech_active = True
                    self.speech_start_time = time.time()
                    continue

                if (time.time() - self.speech_start_time) < cfg.min_speech_duration:
                    continue

                self.is_speech_active = False
                segment_audio = self._extract_segment(
                    audio_concat, start_time, end_time
                )
                asyncio.create_task(self.transcribe_and_speak(segment_audio))
                self.vad_buffer = []

            except Exception:
                print("Error in process_audio_loop:")
                traceback.print_exc()
                await asyncio.sleep(0.1)

    def _extract_segment(
        self, audio_concat: np.ndarray, start_time: float, end_time: float
    ) -> np.ndarray:
        cfg = self.config
        start_sample = max(
            0,
            int(start_time * cfg.sample_rate)
            - int(cfg.buffer_before_speech * cfg.sample_rate),
        )
        end_sample = min(int(end_time * cfg.sample_rate), len(audio_concat))

        if start_sample > 0:
            return audio_concat[start_sample:end_sample]

        silence_audio = np.concatenate(self.silence_buffer)
        silence_needed = min(
            len(silence_audio), int(cfg.buffer_before_speech * cfg.sample_rate)
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

    transformer = VoiceTransformer(config, output_device_idx=output_device_idx)
    transformer.run()


if __name__ == "__main__":
    main()
