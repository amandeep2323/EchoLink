# TTS Voice Model Directory

Place your Piper TTS voice files here. TTS is **optional** — 
sign detection works without it.

## Setup

1. Download a Piper voice from:
   https://github.com/rhasspy/piper/releases

2. You need TWO files per voice:
   ```
   python-backend/models/tts/
   ├── en_US-lessac-medium.onnx          ← Voice model
   └── en_US-lessac-medium.onnx.json     ← Voice config
   ```

3. The default voice name is `en_US-lessac-medium`.
   To use a different voice, change the `tts_voice` setting
   in the frontend Settings panel.

## Recommended Voices

| Voice | Quality | Size | Speed |
|-------|---------|------|-------|
| `en_US-lessac-medium` | Good | ~60MB | Fast |
| `en_US-lessac-high` | Best | ~100MB | Slower |
| `en_US-amy-low` | OK | ~15MB | Fastest |

## What NOT to put here

- ❌ Sign language models — Those go in `models/sign/`
- ❌ labels.json — That goes in `models/sign/`
