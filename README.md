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

## Notes

- The first run downloads the Whisper and Silero models (a few hundred MB).
- End-to-end latency from "you stop speaking" to "you hear the transformed voice" is typically 600–900ms on a GPU machine with `tiny.en` + `flash_v2_5`.
- `test.wav` is a sample input clip for offline testing.
