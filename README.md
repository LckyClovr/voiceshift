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

**Switching voices mid-session:** while VoiceShift is running, type a voice index (e.g. `2`) and press Enter to swap voices for the next utterance — no restart needed. The list of available indices is printed when the app starts.

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
| `USE_REALTIME` | `true` | Use ElevenLabs' websocket realtime API. `false` falls back to HTTP streaming |
| `EDGE_VOICE` | `en-US-GuyNeural` | Used only with Edge TTS |
| `WHISPER_MODEL` | `tiny.en` | `tiny.en` / `base.en` / `small.en` / `medium.en` |
| `WHISPER_BEAM_SIZE` | `5` | 1 = greedy/fastest; 5 = better first-word accuracy |
| `PRE_ROLL_SECONDS` | `0.5` | Audio prepended to each utterance to avoid clipping soft onsets |
| `SILENCE_END_MS` | `350` | Trailing silence required to end an utterance |
| `MME_ONLY` | `false` | Restrict the output device picker to the MME host API |
| `VOCAB_FILE` | `vocab.txt` | Custom vocabulary file (see below) |
| `CARRY_INFLECTION` | `false` | Detect and carry over prosody (see below) |
| `PROSODY_DEBUG` | `false` | Print per-utterance prosody cues and per-word pitch/energy diagnostics |

## Custom vocabulary

VoiceShift loads `vocab.txt` at startup to bias Whisper toward proper nouns, jargon, brand names, or slang it would otherwise mishear. The file is gitignored so each user can keep their own.

Four line types:

- **Bias term** — a plain word/phrase, passed to Whisper's `initial_prompt`. Whisper is more likely to produce this spelling, but it's a soft hint.
- **Hotword** — `! word` applies a stronger logit boost during decoding via faster-whisper's `hotwords` parameter. Use this for slang Whisper still misses after a plain bias entry.
- **Substitution** — `wrong -> right` forces a literal replacement (case-insensitive, whole-word). The right-hand spelling is also added to the bias prompt automatically.
- **Anchored substitution** — `^wrong$ -> right` only fires when `wrong` is the entire transcript (trailing punctuation is allowed). Useful for short standalone words where Whisper has no surrounding context, e.g. "naw" → "now".

```
# vocab.txt
ElevenLabs
! rizz
voice shift -> VoiceShift
^now$ -> nah
```

Lines starting with `#` and blank lines are ignored. On startup you'll see `Loaded vocab from vocab.txt: N bias terms, M substitutions` confirming it loaded. Use `VOCAB_FILE=path/to/file.txt` in `.env` to point at a different file.

Tip: if a term keeps getting mangled even after adding a bias entry, check what Whisper is actually outputting and add a `wrong -> right` substitution for that exact mishear.

## Inflection carryover

Set `CARRY_INFLECTION=true` to have VoiceShift extract simple prosodic features from your audio and inject them into the text sent to TTS so the synthesized voice mirrors your delivery. Adds ~10–30ms per utterance.

What gets carried over (all per-word, using Whisper's word timestamps):

- **Questions** — last word's pitch ≥12% above the median pitch of earlier words adds a `?`. Suppressed if the last word is also loud (more likely an exclamation).
- **Exclamations** — last word's energy ≥1.5× the median earlier energy adds a `!`.
- **Pauses** — gaps >300ms between words become `...`.
- **Emphasis** — words with both elevated energy and elevated duration-per-character relative to this utterance's median get capitalized. Skips one-letter words and the final word (handled by ?/!).
- **Speaking rate** — measured but currently only logged.

This is an *approximation*, not full prosody transfer. You'll get question intonation, exclamations, pauses, and rough emphasis, but not your melodic contour, micro-timing, or vocal texture. For genuine inflection transfer, use ElevenLabs' Speech-to-Speech endpoint instead (separate code path, not yet wired up).

Set `PROSODY_DEBUG=true` to log the detected cues alongside each transcript, e.g.:

```
Transcribed (180ms) [?, emph=1, pauses=2]: are you SERIOUS ... right now?
Transcribed (160ms) [!]: that's INSANE!
```

With `PROSODY_DEBUG=false` (the default), only the final text is printed — the prosody analysis still runs and still shapes the TTS output, just without the diagnostic line.

## Notes

- The first run downloads the Whisper and Silero models (a few hundred MB).
- End-to-end latency from "you stop speaking" to "you hear the transformed voice" is typically 600–900ms on a GPU machine with `tiny.en` + `flash_v2_5`.
- `test.wav` is a sample input clip for offline testing.
