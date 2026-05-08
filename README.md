# VoiceShift

Real-time voice transformer. Speak into your mic and hear your words spoken back in a different voice with sub-second latency.

Microphone audio is gated by Silero VAD, transcribed locally with `faster-whisper`, and re-synthesized through ElevenLabs (or Microsoft Edge TTS as a free fallback). The output is routed to any system audio device — pair it with a virtual cable (e.g. VB-Cable) to feed the transformed voice into a meeting, game, or stream as if it were your microphone.

## How it works

```
mic → InputStream → VAD (Silero) → Whisper STT → TTS (ElevenLabs / Edge) → output device
```

- **Streaming capture** in 0.25s blocks with a 4s rolling buffer.
- **Pre-roll silence buffer** prepended to detected speech so the first phoneme is never clipped.
- **Endpointing on trailing silence** — the moment you stop talking, the pipeline fires. No fixed wait.
- **Streaming TTS + streaming playback** — PCM chunks from ElevenLabs are written straight to the output device as they arrive, so the first audio is heard within ~300ms of synthesis starting (no MP3 decode round-trip, no waiting for the full clip).
- **`eleven_flash_v2_5`** by default — ElevenLabs' lowest-latency model.
- **Single-flight processing** — an `asyncio.Lock` ensures one utterance is fully spoken before the next is picked up.
- **GPU when available**, CPU fallback automatically.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in ELEVEN_API_KEY and your voice library in ELEVEN_VOICES
python voiceshift.py
```

On startup VoiceShift prompts for an output device, then for a voice from the library you defined in `.env`.

For GPU acceleration, install a CUDA-enabled build of PyTorch separately; otherwise it falls back to CPU with `int8` quantization.

## Configuring your voice library

Voices are defined in `.env` as a comma-separated list of `name:voice_id` pairs:

```
ELEVEN_VOICES=Villain:zYcjlYFOd3taleS0gkk3,Old Man:NOpBlnGInO9m6vDvFkFC,British Guy:7S3KNdLDL7aRgBVRQb1z
```

Names are local labels for your own reference; the voice IDs come from your ElevenLabs voice library. The picker uses the order you list them in.

## Configuration

All settings are env vars (see `.env.example`):

| Variable | Default | Notes |
|---|---|---|
| `ELEVEN_API_KEY` | — | Required if `USE_ELEVENLABS=true` |
| `ELEVEN_VOICES` | — | Comma-separated `name:id` pairs; preferred over `ELEVEN_VOICE_ID` |
| `ELEVEN_VOICE_ID` | — | Fallback if `ELEVEN_VOICES` is empty |
| `ELEVEN_MODEL_ID` | `eleven_flash_v2_5` | `flash_v2_5` (fastest), `turbo_v2_5`, or `multilingual_v2` (highest fidelity) |
| `USE_ELEVENLABS` | `true` | Set `false` to use Edge TTS instead |
| `EDGE_VOICE` | `en-US-GuyNeural` | Used only with Edge TTS |
| `WHISPER_MODEL` | `tiny.en` | `tiny.en` / `base.en` / `small.en` / `medium.en` |
| `SILENCE_END_MS` | `350` | Trailing silence required to end an utterance |
| `VOCAB_FILE` | `vocab.txt` | Custom vocabulary file (see below) |
| `CARRY_INFLECTION` | `false` | Detect and carry over prosody (see below) |

## Custom vocabulary

VoiceShift loads `vocab.txt` at startup to bias Whisper toward proper nouns, jargon, brand names, or slang it would otherwise mishear. The file is gitignored so each user can keep their own.

Two line types:

- **Bias term** — a plain word/phrase, passed to Whisper's `initial_prompt`. Whisper is more likely to produce this spelling, but it's a soft hint, not a guarantee.
- **Substitution** — `wrong -> right` forces a literal replacement on the transcript after Whisper runs. Match is case-insensitive and whole-word; replacement preserves your right-hand casing. The right-hand spelling is also added to the bias prompt automatically.

```
# vocab.txt
ElevenLabs
faster-whisper
voice shift -> VoiceShift
11 labs -> ElevenLabs
```

Lines starting with `#` and blank lines are ignored. On startup you'll see `Loaded vocab from vocab.txt: N bias terms, M substitutions` confirming it loaded. Use `VOCAB_FILE=path/to/file.txt` in `.env` to point at a different file.

Tip: if a term keeps getting mangled even after adding a bias entry, check what Whisper is actually outputting and add a `wrong -> right` substitution for that exact mishear.

## Inflection carryover

Set `CARRY_INFLECTION=true` to have VoiceShift extract simple prosodic features from your audio and inject them into the text sent to TTS so the synthesized voice mirrors your delivery. Adds ~10–30ms per utterance.

What gets carried over:

- **Questions** — rising pitch in the last ~25% of the utterance vs the middle adds a `?`.
- **Pauses** — gaps >250ms between words become `...` so the TTS pauses there too.
- **Emphasis** — words >1.5σ above mean RMS energy get capitalized so the TTS stresses them.
- **Speaking rate** — measured but currently only logged; reserved for SSML wiring.

This is an *approximation*, not full prosody transfer. You'll get question intonation, pauses, and rough emphasis, but not your melodic contour, micro-timing, or vocal texture. For genuine inflection transfer, use ElevenLabs' Speech-to-Speech endpoint instead (separate code path, not yet wired up).

When on, transcripts log with the detected cues, e.g.:

```
Transcribed (180ms) [?, emph=1, pauses=2]: are you SERIOUS ... right now?
```

## Notes

- The first run downloads the Whisper and Silero models (a few hundred MB).
- End-to-end latency from "you stop speaking" to "you hear the transformed voice" is typically 600–900ms on a GPU machine with `tiny.en` + `flash_v2_5`.
- `test.wav` is a sample input clip for offline testing.
