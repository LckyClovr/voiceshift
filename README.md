# VoiceShift

Real-time voice transformer. Speak into your mic and hear your words spoken back in a different voice with sub-second latency.

Microphone audio is gated by Silero VAD, transcribed locally with `faster-whisper`, and re-synthesized through ElevenLabs (or Microsoft Edge TTS as a free fallback). The output is routed to any system audio device — pair it with a virtual cable (e.g. VB-Cable) to feed the transformed voice into a meeting, game, or stream as if it were your microphone.

## How it works

```
mic → InputStream → VAD (Silero) → Whisper STT → TTS (ElevenLabs / Edge) → output device
```

- **Streaming capture** in 0.25s blocks with a 4s rolling buffer.
- **Pre-roll silence buffer** (0.75s) prepended to detected speech so the first phoneme is never clipped.
- **Single-flight processing** — an `asyncio.Lock` ensures one utterance is fully synthesized and played before the next one is picked up.
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
| `ELEVEN_MODEL_ID` | `eleven_multilingual_v2` | |
| `USE_ELEVENLABS` | `true` | Set `false` to use Edge TTS instead |
| `EDGE_VOICE` | `en-US-GuyNeural` | Used only with Edge TTS |
| `WHISPER_MODEL` | `tiny.en` | `tiny.en` / `base.en` / `small.en` / `medium.en` |

## Notes

- The first run downloads the Whisper and Silero models (a few hundred MB).
- Latency is dominated by the TTS round-trip; ElevenLabs streaming endpoints are a natural next step for lower end-to-end delay.
- `test.wav` is a sample input clip for offline testing.
