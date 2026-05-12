import asyncio
import io
import os
import queue
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

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
class Vocab:
    # Plain words/phrases passed to Whisper's initial_prompt to bias decoding
    # toward this vocabulary.
    bias_terms: list[str] = field(default_factory=list)
    # Strong-bias terms passed to faster-whisper's hotwords param. These get
    # a logit boost during decoding — stronger than initial_prompt alone.
    hotwords: list[str] = field(default_factory=list)
    # (compiled_pattern, replacement) pairs applied to the transcript after
    # Whisper runs. Pattern is case-insensitive whole-word(s) match.
    substitutions: list[tuple[re.Pattern, str]] = field(default_factory=list)

    @property
    def initial_prompt(self) -> str:
        return ", ".join(self.bias_terms) if self.bias_terms else ""

    @property
    def hotwords_str(self) -> str:
        return " ".join(self.hotwords) if self.hotwords else ""


@dataclass
class Config:
    sample_rate: int = 16000
    block_duration: float = 0.25
    rolling_buffer_seconds: int = 4
    vad_threshold: float = 0.4
    min_speech_duration: float = 0.2
    buffer_before_speech: float = 0.5
    silence_end_ms: int = 350
    whisper_model: str = "tiny.en"
    whisper_beam_size: int = 5
    edge_voice: str = "en-US-GuyNeural"
    use_elevenlabs: bool = True
    use_realtime: bool = True
    eleven_voice_id: str = ""
    eleven_model_id: str = "eleven_flash_v2_5"
    eleven_api_key: str = ""
    vocab_file: str = "vocab.txt"
    carry_inflection: bool = False
    prosody_debug: bool = False
    mme_only: bool = False
    voices: list[Voice] = field(default_factory=list)
    vocab: Vocab = field(default_factory=Vocab)


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


def load_vocab(path: str) -> Vocab:
    """Parse vocab.txt. Supported line formats:

    - `word`               — bias term (added to Whisper's initial_prompt)
    - `! word`             — hotword (stronger bias via faster-whisper hotwords)
    - `wrong -> right`     — case-insensitive whole-word substitution
    - `^wrong$ -> right`   — anchored: only fires if `wrong` is the entire
                              transcript (trailing punctuation ignored). Useful
                              for short utterances where context is missing.
    """
    p = Path(path)
    if not p.exists():
        return Vocab()

    bias_terms: list[str] = []
    hotwords: list[str] = []
    substitutions: list[tuple[re.Pattern, str]] = []
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "->" in line:
            wrong, right = line.split("->", 1)
            wrong, right = wrong.strip(), right.strip()
            if not wrong or not right:
                continue
            anchored = wrong.startswith("^") and wrong.endswith("$")
            if anchored:
                inner = wrong[1:-1].strip()
                if not inner:
                    continue
                # Match the entire string (allowing trailing punctuation
                # like "Now." or "now?"), case-insensitive.
                pattern = re.compile(
                    rf"^\s*{re.escape(inner)}\s*[.,!?;:]?\s*$", re.IGNORECASE
                )
            else:
                # Whole-token match so "AI" doesn't replace inside "again".
                pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
            substitutions.append((pattern, right))
            # Also seed the bias prompt with the right-hand spelling so
            # Whisper has a chance to produce it directly.
            bias_terms.append(right)
        elif line.startswith("!"):
            term = line[1:].strip()
            if term:
                hotwords.append(term)
                bias_terms.append(term)
        else:
            bias_terms.append(line)

    return Vocab(
        bias_terms=bias_terms, hotwords=hotwords, substitutions=substitutions
    )


def load_config() -> Config:
    load_dotenv()
    vocab_file = os.environ.get("VOCAB_FILE", "vocab.txt")
    return Config(
        whisper_model=os.environ.get("WHISPER_MODEL", "tiny.en"),
        edge_voice=os.environ.get("EDGE_VOICE", "en-US-GuyNeural"),
        use_elevenlabs=os.environ.get("USE_ELEVENLABS", "true").lower() == "true",
        use_realtime=os.environ.get("USE_REALTIME", "true").lower() == "true",
        eleven_voice_id=os.environ.get("ELEVEN_VOICE_ID", ""),
        eleven_model_id=os.environ.get("ELEVEN_MODEL_ID", "eleven_flash_v2_5"),
        eleven_api_key=os.environ.get("ELEVEN_API_KEY", ""),
        voices=parse_voices(os.environ.get("ELEVEN_VOICES", "")),
        silence_end_ms=int(os.environ.get("SILENCE_END_MS", "350")),
        buffer_before_speech=float(os.environ.get("PRE_ROLL_SECONDS", "0.5")),
        whisper_beam_size=int(os.environ.get("WHISPER_BEAM_SIZE", "5")),
        vocab_file=vocab_file,
        vocab=load_vocab(vocab_file),
        carry_inflection=os.environ.get("CARRY_INFLECTION", "false").lower() == "true",
        prosody_debug=os.environ.get("PROSODY_DEBUG", "false").lower() == "true",
        mme_only=os.environ.get("MME_ONLY", "false").lower() == "true",
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


def estimate_pitch_hz(audio: np.ndarray, sr: int) -> float:
    """Autocorrelation-based pitch estimate. Returns 0.0 if unvoiced.

    Cheap and good enough for "did pitch rise at the end of the utterance" —
    not for music. Searches the typical human speech range (75–400 Hz).
    """
    if len(audio) < sr // 20:
        return 0.0
    audio = audio - audio.mean()
    if audio.std() < 0.01:
        return 0.0
    min_lag = sr // 400
    max_lag = sr // 75
    if max_lag >= len(audio):
        return 0.0
    corr = np.correlate(audio, audio, mode="full")[len(audio) - 1 :]
    corr = corr[min_lag:max_lag]
    if corr.size == 0 or corr.max() <= 0:
        return 0.0
    peak_lag = int(np.argmax(corr)) + min_lag
    return float(sr) / peak_lag


@dataclass
class Prosody:
    is_question: bool = False
    is_exclamation: bool = False
    pause_indices: list[int] = field(default_factory=list)  # word indices that get a pause inserted *before* them
    emphasis_indices: set[int] = field(default_factory=set)  # word indices to emphasize
    rate: float = 1.0  # 1.0 = normal, <1 = slower, >1 = faster


def _word_pitch(
    audio: np.ndarray, sr: int, start: float, end: float
) -> float:
    """Pitch (Hz) over a word span, or 0 if unvoiced/too short."""
    s = max(0, int(start * sr))
    e = min(len(audio), int(end * sr))
    # Need at least ~50ms of signal for a stable estimate.
    if e - s < sr // 20:
        return 0.0
    return estimate_pitch_hz(audio[s:e], sr)


def _word_energy(
    audio: np.ndarray, sr: int, start: float, end: float
) -> float:
    """RMS energy over a word span."""
    s = max(0, int(start * sr))
    e = min(len(audio), int(end * sr))
    if e <= s:
        return 0.0
    seg = audio[s:e]
    return float(np.sqrt(np.mean(seg * seg)))


# Function words systematically have lower energy and shorter duration than
# content words. Excluding them from the baseline keeps "REALLY" from being
# compared against "is" and "the" — which would make every content word
# look like an outlier and dilute real emphasis.
_FUNCTION_WORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "am", "do", "does", "did", "have", "has", "had", "of", "to", "in",
        "on", "at", "for", "with", "from", "by", "as", "and", "or", "but",
        "if", "so", "it", "its", "this", "that", "these", "those", "i",
        "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "their", "our", "no", "not",
    }
)


def _normalize_word(w: str) -> str:
    return "".join(c for c in w.lower() if c.isalpha())


def analyze_prosody(
    audio: np.ndarray,
    sr: int,
    word_timings: list[tuple[str, float, float]],
    debug: bool = False,
) -> Prosody:
    """Extract prosodic features from the audio + Whisper word timings.

    word_timings is a list of (word, start_seconds, end_seconds).
    """
    prosody = Prosody()
    if not word_timings or len(audio) < sr // 4:
        return prosody

    pitches = [_word_pitch(audio, sr, s, e) for _, s, e in word_timings]
    energies = [_word_energy(audio, sr, s, e) for _, s, e in word_timings]
    dpc = []
    for word, start, end in word_timings:
        chars = max(1, len(word.strip()))
        dpc.append((end - start) / chars)

    last = len(word_timings) - 1
    norm_words = [_normalize_word(w) for w, _, _ in word_timings]

    # Indices of "content words" — used as the baseline cohort. Function
    # words are excluded because they're systematically quieter/shorter
    # and would skew the median downward.
    content_idx = [
        i
        for i, w in enumerate(norm_words)
        if w and w not in _FUNCTION_WORDS and len(w) > 1
    ]

    # 1. Question vs exclamation, computed against the content-word baseline.
    # Use the *max* of earlier content words, not the median, so natural
    # final-word stress in a declarative ("really COOL") doesn't trigger
    # a question — a real question keeps rising past where the speaker
    # already pushed pitch on emphasized content.
    if last >= 1:
        prior_content_pitches = [
            pitches[i] for i in content_idx if i < last and pitches[i] > 50
        ]
        prior_content_energies = [
            energies[i] for i in content_idx if i < last and energies[i] > 1e-5
        ]
        last_pitch = pitches[last]
        last_energy = energies[last]

        if prior_content_pitches and prior_content_energies and last_pitch > 50:
            max_prior_pitch = float(max(prior_content_pitches))
            median_prior_energy = float(np.median(prior_content_energies))

            # Exclamation: last word is much louder than every prior
            # content word AND its pitch is at or above the prior content
            # max. The pitch condition rules out the common "structural
            # final-word loudness" case in declaratives, where the last
            # word is naturally louder but pitch falls.
            louder_than_all_content = all(
                last_energy >= energies[i] * 1.5
                for i in content_idx
                if i < last
            )
            pitch_not_falling = (
                max_prior_pitch <= 0 or last_pitch >= max_prior_pitch * 0.95
            )
            energy_ratio = (
                last_energy / median_prior_energy if median_prior_energy > 0 else 1.0
            )
            if (
                energy_ratio >= 1.8
                and last_energy > 0.03
                and louder_than_all_content
                and pitch_not_falling
            ):
                prosody.is_exclamation = True

            # Question: last word's pitch exceeds the highest pitch of any
            # earlier content word by a clear margin. Requires ≥2 prior
            # content-word pitches to compare against — a single prior
            # word can't tell us whether a pitch jump is "question rise"
            # or just "natural stress on the final content word in a
            # declarative." When in doubt, don't flag it.
            if (
                len(prior_content_pitches) >= 2
                and last_pitch >= max_prior_pitch * 1.15
                and not prosody.is_exclamation
            ):
                prosody.is_question = True

    # 2. Emphasis: a content word that stands out from *other* content
    # words in pitch, energy, or duration. Pitch is the strongest cue —
    # speakers often emphasize via pitch alone, with energy/duration
    # comparable to surrounding words. We score each word against the
    # median of the others so a 2-content-word utterance like "really
    # cool" can still flag one as emphasized.
    #
    # Sentence-final words are universally longer and slightly louder in
    # English (final lengthening + utterance-final intensity), so for the
    # last word we require stricter thresholds and disable the duration
    # cue entirely — otherwise we'd flag the final word in nearly every
    # declarative as "emphasized."
    if len(content_idx) >= 2:
        e_arr = np.array(energies)
        d_arr = np.array(dpc)
        p_arr = np.array(pitches)

        for i in content_idx:
            if i == last and (prosody.is_question or prosody.is_exclamation):
                continue
            others = [j for j in content_idx if j != i]
            if not others:
                continue
            e_med = float(np.median([e_arr[j] for j in others]))
            d_med = float(np.median([d_arr[j] for j in others]))
            other_pitches = [p_arr[j] for j in others if p_arr[j] > 50]
            p_max = float(max(other_pitches)) if other_pitches else 0.0

            is_final = i == last
            # Pitch elevation: works for both mid- and final-word emphasis.
            pitch_high = (
                p_arr[i] > 50 and p_max > 50 and p_arr[i] >= p_max * 1.40
            )
            # Energy: stricter threshold for the final word since it's
            # naturally a bit louder.
            energy_thresh = 1.45 if is_final else 1.30
            energy_high = e_arr[i] >= e_med * energy_thresh and e_arr[i] > 0.012
            # Duration: only cue mid-utterance words; final-word lengthening
            # is structural, not emphatic.
            duration_high = (
                not is_final and d_arr[i] >= d_med * 1.25
            )
            if pitch_high or energy_high or duration_high:
                prosody.emphasis_indices.add(i)

    # 3. Pauses.
    for i in range(1, len(word_timings)):
        gap = word_timings[i][1] - word_timings[i - 1][2]
        if gap >= 0.30:
            prosody.pause_indices.append(i)

    # 4. Speaking rate.
    total_chars = sum(len(w[0]) for w in word_timings)
    total_sec = word_timings[-1][2] - word_timings[0][1]
    if total_sec > 0.3 and total_chars > 0:
        chps = total_chars / total_sec
        prosody.rate = max(0.7, min(1.3, chps / 15.0))

    if debug:
        print("  prosody-debug:")
        for i, ((word, st, en), p, e, d) in enumerate(
            zip(word_timings, pitches, energies, dpc)
        ):
            tags = []
            if i in prosody.emphasis_indices:
                tags.append("EMPH")
            if i == last and prosody.is_question:
                tags.append("Q")
            if i == last and prosody.is_exclamation:
                tags.append("EXC")
            tag_str = f" [{','.join(tags)}]" if tags else ""
            kind = "fn" if norm_words[i] in _FUNCTION_WORDS else "ct"
            print(
                f"    {i:>2} {kind} {word!r:>14} "
                f"dur={en - st:.2f}s pitch={p:5.1f}Hz "
                f"rms={e:.4f} d/c={d:.3f}{tag_str}"
            )

    return prosody


def render_with_prosody(
    word_timings: list[tuple[str, float, float]],
    prosody: Prosody,
) -> str:
    """Reassemble the transcript with prosodic cues embedded as punctuation
    and capitalization. We avoid SSML because Flash v2.5 only partially
    honors it — punctuation and casing are universally respected by TTS."""
    if not word_timings:
        return ""
    parts: list[str] = []
    for i, (word, _, _) in enumerate(word_timings):
        token = word.strip()
        if not token:
            continue
        if i in prosody.pause_indices and parts:
            parts.append("...")
        if i in prosody.emphasis_indices:
            # Capitalize the alphabetic part to cue emphasis without breaking
            # punctuation already attached by Whisper.
            token = "".join(c.upper() if c.isalpha() else c for c in token)
        parts.append(token)
    text = " ".join(parts)

    # Terminal punctuation: question wins over exclamation only if both
    # were detected, but exclamation should already have suppressed
    # is_question in the analyzer.
    if text:
        if prosody.is_question and text[-1] not in "?!.":
            text = text.rstrip(",;:") + "?"
        elif prosody.is_exclamation and text[-1] not in "?!.":
            text = text.rstrip(",;:") + "!"
    return text


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
        if (
            config.vocab.bias_terms
            or config.vocab.substitutions
            or config.vocab.hotwords
        ):
            print(
                f"Loaded vocab from {config.vocab_file}: "
                f"{len(config.vocab.bias_terms)} bias terms, "
                f"{len(config.vocab.hotwords)} hotwords, "
                f"{len(config.vocab.substitutions)} substitutions"
            )
        if config.carry_inflection:
            print("Inflection carryover: ON (questions, pauses, emphasis)")
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

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
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
            if self.config.use_realtime:
                # Websocket-based realtime API. text is an iterator so the
                # SDK can stream tokens — we send the full utterance as a
                # single chunk and close.
                pcm_iter = self.eleven_client.text_to_speech.convert_realtime(
                    self.voice_id,
                    text=iter([text]),
                    model_id=self.config.eleven_model_id,
                    output_format=ELEVEN_PCM_FORMAT,
                    voice_settings=None,
                )
            else:
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
                vocab = self.config.vocab
                carry = self.config.carry_inflection
                transcribe_kwargs = {"beam_size": self.config.whisper_beam_size}
                if vocab.initial_prompt:
                    transcribe_kwargs["initial_prompt"] = vocab.initial_prompt
                if vocab.hotwords_str:
                    transcribe_kwargs["hotwords"] = vocab.hotwords_str
                if carry:
                    transcribe_kwargs["word_timestamps"] = True
                segments, _ = self.whisper.transcribe(audio_np, **transcribe_kwargs)

                if carry:
                    word_timings: list[tuple[str, float, float]] = []
                    for seg in segments:
                        for w in (seg.words or []):
                            word_timings.append((w.word, w.start, w.end))
                    prosody = analyze_prosody(
                        audio_np,
                        self.config.sample_rate,
                        word_timings,
                        debug=self.config.prosody_debug,
                    )
                    text = render_with_prosody(word_timings, prosody)
                else:
                    text = " ".join(seg.text.strip() for seg in segments).strip()

                for pattern, replacement in vocab.substitutions:
                    text = pattern.sub(replacement, text)
                t_stt = time.time() - t0
                if not text:
                    return

                if carry and self.config.prosody_debug:
                    cues = []
                    if prosody.is_question:
                        cues.append("?")
                    if prosody.is_exclamation:
                        cues.append("!")
                    if prosody.emphasis_indices:
                        cues.append(f"emph={len(prosody.emphasis_indices)}")
                    if prosody.pause_indices:
                        cues.append(f"pauses={len(prosody.pause_indices)}")
                    if abs(prosody.rate - 1.0) > 0.05:
                        cues.append(f"rate={prosody.rate:.2f}")
                    cue_str = f" [{', '.join(cues)}]" if cues else ""
                    print(f"Transcribed ({t_stt * 1000:.0f}ms){cue_str}: {text}")
                else:
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

    def switch_voice(self, idx: int) -> None:
        voices = self.config.voices
        if not (0 <= idx < len(voices)):
            print(f"  (no voice at index {idx}; have 0–{len(voices) - 1})")
            return
        chosen = voices[idx]
        if chosen.voice_id == self.voice_id:
            print(f"  (already on {chosen.name})")
            return
        self.voice_id = chosen.voice_id
        print(f"  -> switched to voice {idx}: {chosen.name}")

    def voice_switch_thread(self) -> None:
        voices = self.config.voices
        if not voices:
            return
        print("\nVoice switch: type a number + Enter to change voice mid-session.")
        for i, v in enumerate(voices):
            print(f"  {i}: {v.name}")
        print()
        while True:
            try:
                raw = input().strip()
            except EOFError:
                return
            if not raw:
                continue
            try:
                self.switch_voice(int(raw))
            except ValueError:
                print(f"  (not a number: {raw!r})")

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
        if self.config.use_elevenlabs and len(self.config.voices) > 1:
            threading.Thread(target=self.voice_switch_thread, daemon=True).start()
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


def list_output_devices(mme_only: bool = False) -> list[tuple[int, str]]:
    """Dedupe physical devices that show up once per host API on Windows.

    sd.query_devices() returns the same speaker under MME, DirectSound,
    WASAPI, and WDM-KS — only one of those typically streams PCM cleanly
    at our sample rate. We pick one entry per device name, preferring
    WASAPI > DirectSound > MME > anything else.

    Pass mme_only=True to restrict to the MME host API only — useful when
    other host APIs misbehave on your system.
    """
    host_apis = sd.query_hostapis()
    priority = {"Windows WASAPI": 0, "Windows DirectSound": 1, "MME": 2}

    best_per_name: dict[str, tuple[int, int, str]] = {}  # name -> (priority, idx, hostapi_name)
    for idx, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] <= 0:
            continue
        hostapi_name = host_apis[d["hostapi"]]["name"]
        if mme_only and hostapi_name != "MME":
            continue
        rank = priority.get(hostapi_name, 99)
        existing = best_per_name.get(d["name"])
        if existing is None or rank < existing[0]:
            best_per_name[d["name"]] = (rank, idx, hostapi_name)

    return [
        (idx, f"{name}  [{hostapi_name}]")
        for name, (_, idx, hostapi_name) in best_per_name.items()
    ]


def main() -> None:
    config = load_config()

    devices = list_output_devices(mme_only=config.mme_only)
    print("Available output devices:")
    for display_idx, (_, name) in enumerate(devices):
        print(f"  {display_idx}: {name}")
    raw = input("Choose output device index (blank for system default): ").strip()
    if raw:
        choice = int(raw)
        if not 0 <= choice < len(devices):
            raise SystemExit(f"Index out of range. Pick 0–{len(devices) - 1}.")
        output_device_idx = devices[choice][0]
    else:
        output_device_idx = None

    voice_id = pick_voice(config) if config.use_elevenlabs else ""

    transformer = VoiceTransformer(
        config, voice_id=voice_id, output_device_idx=output_device_idx
    )
    transformer.run()


if __name__ == "__main__":
    main()
