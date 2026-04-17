# SignSpeak — Python Backend

FastAPI WebSocket server that powers the SignSpeak ASL-to-Speech application.

## Architecture

```
python-backend/
├── main.py                          ← Entry point (starts uvicorn on port 8765)
├── requirements.txt                 ← Python dependencies
├── README.md                        ← You are here
└── src/
    ├── __init__.py
    │
    ├── server/                      ← WebSocket server + protocol ✅
    │   ├── __init__.py
    │   ├── app.py                   ← FastAPI app factory, health, CORS, lifespan
    │   ├── protocol.py              ← Message types (enums) + JSON message builders
    │   ├── connection_manager.py    ← WebSocket connection tracking, broadcast/send
    │   ├── websocket_handler.py     ← WS route handler, message routing
    │   └── pipeline_manager.py      ← Pipeline state, loop, device enumeration
    │
    ├── models/                      ← Model auto-detection + loading ✅
    │   ├── __init__.py
    │   ├── model_loader.py          ← Unified loader (auto-detect → ONNX Runtime)
    │   ├── converter.py             ← .h5/.keras/.tflite → .onnx conversion
    │   └── label_map.py             ← Label loading (JSON, TXT, default A-Z)
    │
    ├── camera/                      ← Camera capture + compositing + VCam ✅
    │   ├── __init__.py
    │   ├── capture.py               ← Threaded OpenCV camera capture
    │   ├── compositor.py            ← Video overlay engine (transcript, signs, landmarks)
    │   └── virtual_camera.py        ← pyvirtualcam → OBS Virtual Camera output
    │
    ├── speech/                      ← TTS engine + virtual mic ✅
    │   ├── __init__.py
    │   ├── tts_engine.py            ← Piper TTS (offline, neural, threaded queue)
    │   └── virtual_mic.py           ← sounddevice → VB-Audio Virtual Cable
    │
    └── recognition/                 ← (Phase 4) ML pipeline: landmarks → signs → text
```

## Setup

```bash
cd python-backend
python -m venv venv

# Activate virtual environment
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux/macOS

pip install -r requirements.txt
```

### Model Conversion Dependencies (one-time)

If your model is in `.h5`, `.keras`, or `.tflite` format, install conversion tools:

```bash
pip install tensorflow tf2onnx
```

The converter will auto-cache the `.onnx` file next to the original, so this only runs once.

## Running

```bash
python main.py
```

## Module Documentation

### `src/models/` — Model Auto-Detection System

The model loader automatically detects and handles 4 formats:

| Format   | Extension  | Handling                                     |
|----------|------------|----------------------------------------------|
| ONNX     | `.onnx`    | Loaded directly with ONNX Runtime            |
| Keras    | `.h5`      | Converted to `.onnx` via tf2onnx, then loaded |
| Keras    | `.keras`   | Converted to `.onnx` via tf2onnx, then loaded |
| TFLite   | `.tflite`  | Converted to `.onnx` via tf2onnx, then loaded |

**Key features:**
- **Auto-discovery**: Scans a directory for model files (priority: .onnx > .h5 > .keras > .tflite)
- **Cached conversion**: Converted `.onnx` files are saved alongside the original — only converts once
- **Stale cache detection**: Re-converts if the source model is newer than the cached `.onnx`
- **Label map loading**: Auto-discovers `labels.json`, `labels.txt`, etc. or falls back to A-Z
- **Label validation**: Warns if model output dimensions don't match label count
- **GPU support**: Optional CUDA acceleration via ONNX Runtime GPU provider
- **Unified API**: `predict_sign(features)` returns `(sign, confidence, top_3)` regardless of source format

```python
from src.models import ModelLoader

loader = ModelLoader()
loader.load("path/to/model.h5")   # auto-converts to ONNX, loads
loader.load("path/to/model.onnx") # loads directly

sign, confidence, top_3 = loader.predict_sign(features)
```

### `src/camera/` — Camera Module

#### `capture.py` — Threaded Camera Capture
- Background thread reads frames from OpenCV at target FPS
- Bounded frame queue (drops old frames if consumer is slow)
- Camera watchdog detects disconnection (no frames for 10s)
- Configurable resolution, FPS, and horizontal mirroring
- Base64 JPEG encoding for WebSocket transport

#### `compositor.py` — Video Overlay Engine
- **Transcript bar**: Semi-transparent bar at bottom with scrolling text
- **Sign detection box**: Top-right box showing current sign + confidence bar
- **Status dot**: Top-left indicator (green = hands detected, amber = idle)
- **Hand landmarks**: MediaPipe hand connections + points (left=violet, right=green)
- **Pose landmarks**: Upper body skeleton (shoulders, elbows, wrists)
- All overlays are optional and individually toggleable

#### `virtual_camera.py` — OBS Virtual Camera Output
- Sends composited BGR frames to OBS Virtual Camera via pyvirtualcam
- Auto-resizes frames if resolution doesn't match
- Requires OBS Studio installed for driver registration

### `src/speech/` — Speech Module

#### `tts_engine.py` — Piper TTS Engine
- **Offline neural TTS**: No cloud API, runs entirely locally
- **Synchronous mode**: `synthesize("text")` → numpy int16 array
- **Async mode**: `speak("text")` queues text, background thread synthesizes, callback delivers audio
- **Voice model loading**: Auto-finds `.onnx` + `.onnx.json` voice files
- WAV output → numpy conversion for direct audio processing

#### `virtual_mic.py` — VB-Audio Virtual Cable Output
- Outputs TTS audio through a virtual audio device
- **Device auto-detection**: `find_virtual_cable()` locates VB-Cable Input
- **Non-blocking playback**: Queue-based, background thread streams audio
- **Blocking playback**: `play_blocking()` for synchronous use
- **Resampling**: Linear interpolation when TTS sample rate doesn't match device rate
- Chunk-based streaming for smooth audio output

## Model Formats — Decision Flow

```
User has model file
       │
       ▼
 ┌─ .onnx? ──→ Load directly with ONNX Runtime ──→ ✅ Ready
 │
 ├─ .h5? ────→ Check for cached .onnx ─┬─ Found → Load cached .onnx ──→ ✅ Ready
 │                                      └─ Not found → Convert with tf2onnx
 │                                                     Save .onnx alongside .h5
 │                                                     Load .onnx ──→ ✅ Ready
 │
 ├─ .keras? ─→ (same as .h5 flow)
 │
 └─ .tflite? → Check for cached .onnx ─┬─ Found → Load cached .onnx ──→ ✅ Ready
                                        └─ Not found → Convert with tf2onnx
                                                       Save .onnx alongside .tflite
                                                       Load .onnx ──→ ✅ Ready
```

## Prerequisites

### Required Software
- **OBS Studio** — Virtual camera driver ([download](https://obsproject.com/))
- **VB-Audio Virtual Cable** — Virtual mic device ([download](https://vb-audio.com/Cable/))

### Required Models
- **Sign language model** (`.h5`, `.keras`, `.tflite`, or `.onnx`) in `src/models/`
- **Piper voice model** (`.onnx` + `.onnx.json`) in `src/models/` ([download](https://github.com/rhasspy/piper/releases))
- **Label map** (`labels.json` or `labels.txt`) in `src/models/` (optional — defaults to A-Z)

## Endpoints

| Endpoint | Type | Description |
|----------|------|-------------|
| `ws://127.0.0.1:8765/ws` | WebSocket | Main communication channel |
| `http://127.0.0.1:8765/health` | GET | Health check (JSON status) |
