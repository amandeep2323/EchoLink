# EchoLink — Python Backend

FastAPI WebSocket server that powers the EchoLink ASL-to-Speech application.

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
    │   ├── model_loader.py          ← Unified loader (OpenVINO IR → ONNX Runtime fallback)
    │   ├── model_config.py          ← model.json schema + validation (Phase 6)
    │   ├── model_registry.py        ← Model discovery, switching, active model tracking
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
    │   ├── tts_engine.py            ← Multi-backend TTS (Intel OPEA / Piper / pyttsx3)
    │   ├── opea_tts/                ← Intel OPEA SpeechT5 with OpenVINO IR acceleration
    │   │   ├── __init__.py
    │   │   ├── synthesizer.py       ← TTSEngine-compatible wrapper
    │   │   ├── speecht5_core.py     ← Model loading + synthesis (IR or PyTorch)
    │   │   ├── backend_detector.py  ← OpenVINO availability detection
    │   │   └── model_downloader.py  ← HuggingFace auto-download
    │   └── virtual_mic.py           ← sounddevice → VB-Audio Virtual Cable
    │
    └── recognition/                 ← ML pipeline: landmarks → signs → text
        ├── __init__.py
        ├── landmarker.py            ← MediaPipe Hands or Holistic landmark extraction
        ├── recognizer.py            ← Classification, confidence smoothing, accumulation
        └── spell_corrector.py       ← Spell correction for completed words
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

## Packaging (PyInstaller)

```bash
pyinstaller --noconfirm --clean --distpath python-backend/dist --workpath python-backend/build python-backend/echolink-backend.spec
```

Output executable:
- `python-backend/dist/echolink-backend/echolink-backend.exe`

## Module Documentation

### `src/models/` — Model Auto-Detection System

The model loader uses a hybrid inference architecture:

| Format   | Extension  | Handling                                              |
|----------|------------|-------------------------------------------------------|
| OpenVINO IR | `.xml`+`.bin` | Loaded directly with OpenVINO Runtime (fastest) |
| ONNX     | `.onnx`    | OpenVINO if compatible, else ONNX Runtime fallback    |
| Keras    | `.h5`      | Converted to `.onnx` via tf2onnx, then loaded         |
| Keras    | `.keras`   | Converted to `.onnx` via tf2onnx, then loaded         |
| TFLite   | `.tflite`  | Converted to `.onnx` via tf2onnx, then loaded         |

**Inference backends (auto-selected):**
- **Model 1 (PointNet)** → OpenVINO IR (Intel CPU optimized, LATENCY hint)
- **Model 2 (WLASL Pose-TGCN)** → OpenVINO IR (Intel CPU optimized, LATENCY hint)
- **Model 3 (LSTM)** → ONNX Runtime (Loop operator not supported by OpenVINO)

**Key features:**
- **Auto-discovery**: Scans a directory for model files (priority: .xml > .onnx > .h5 > .keras > .tflite)
- **OpenVINO IR preference**: If `model.xml` exists beside `model.onnx`, loads the IR directly
- **Model caching**: OpenVINO compiled models cached at `cache/openvino/` for fast warm starts
- **Cached conversion**: Converted `.onnx` files are saved alongside the original — only converts once
- **Label map loading**: Auto-discovers `labels.json`, `labels.txt`, etc. or falls back to A-Z
- **Label validation**: Warns if model output dimensions don't match label count
- **Unified API**: `predict_sign(features)` returns `(sign, confidence, top_3)` regardless of source format

```python
from src.models import ModelLoader

loader = ModelLoader()
loader.load_from_config(config)   # Picks best backend (OpenVINO IR → ONNX → Keras)

sign, confidence, top_3 = loader.predict_sign(features)
print(loader.backend)  # "openvino", "onnx", or "keras"
```

**Converting models to OpenVINO IR:**
```bash
cd python-backend/models/sign
convert_to_ir.bat          # Converts Model 1 & 2 to IR, skips incompatible Model 3
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
- **Pose landmarks**: Upper body skeleton (shoulders, elbows, wrists) — drawn when using Holistic
- All overlays are optional and individually toggleable

#### `virtual_camera.py` — OBS Virtual Camera Output
- Sends composited BGR frames to OBS Virtual Camera via pyvirtualcam
- Auto-resizes frames if resolution doesn't match
- Requires OBS Studio installed for driver registration

### `src/speech/` — Speech Module

#### `tts_engine.py` — Multi-Backend TTS Engine

Backend priority (auto mode):
1. **Intel OPEA SpeechT5** — OpenVINO IR accelerated, offline, natural voice (primary)
2. **Piper** — Offline, neural, natural voice (fallback)
3. **pyttsx3** — Windows SAPI voices (last resort)

- **Offline inference**: No cloud API, runs entirely locally
- **OpenVINO IR acceleration**: Uses pre-compiled IR at `models/tts/speecht5_openvino/` with persistent compile cache at `cache/openvino_tts/`
- **Synchronous mode**: `synthesize("text")` → numpy int16 array
- **Async mode**: `speak("text")` queues text, background thread synthesizes, callback delivers audio
- **Auto-settings sync**: Frontend persisted settings (TTS on/off) automatically applied at startup

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

### `src/recognition/` — Recognition Pipeline

#### `landmarker.py` — MediaPipe Landmark Extraction
- **Config-driven**: Switches between `mediapipe_hands` and `mediapipe_holistic` based on `model.json`
- **Hands mode**: Single hand, 21 landmarks × (x, y, z) for fingerspelling models
- **Holistic mode**: feature modes — 55-pt WLASL, `signbart_holistic75` (33 pose + 21 + 21 hands), or `holistic_543x3` (full)
- **Normalization**: Min-max, wrist-relative, frame, or none (configurable)
- **Drawing**: Pose skeleton + hand connections rendered on frame

#### `recognizer.py` — Sign Classification + Post-Processing
- **Dual inference modes**:
  - `single_frame`: Classifies each frame independently (PointNet fingerspelling)
  - `sequence`: Buffers frames into a rolling window, classifies full sequence (WLASL word-level)
- **Sequence buffering**: `deque`-based, configurable length from `model.json`
- **Confidence smoothing**: Rolling average per predicted class
- **Letter accumulation**: Stability gates, cooldown, progressive acceptance (fingerspelling)
- **Word emission**: Direct word output with buffer clearing (word-level)
- **Spell correction**: Optional, applied when words are finalized

#### `spell_corrector.py` — Spell Correction
- Corrects finalized words using fuzzy matching
- Enabled/disabled per model via `model.json`

## Model Architecture

The pipeline adapts automatically based on each model's `model.json` configuration:

| Property | Model 1 (PointNet) | Model 2 (SignBart) | Model 3 (LSTM) |
|----------|-------------------|------------------|----------------|
| **Type** | Fingerspelling | Word-level | Word-level |
| **Landmarks** | MediaPipe Hands (21 pts) | MediaPipe Holistic (75 pts) | MediaPipe Holistic (543 pts) |
| **Inference** | Single frame | 48-frame sequence | 30-frame sequence |
| **Input Shape** | `[1, 21, 3]` | `[1, T, 75, 2]` | `[30, 543, 3]` |
| **Output** | 24 letters (A-Y) | 1000 words | 250 signs |
| **Runtime** | OpenVINO IR | SignBart (ONNX, dual-input) | ONNX Runtime |
| **Post-Processing** | Misrecognition fixes + spell correction | Confidence smoothing only | Confidence smoothing only |

## Prerequisites

### Required Software
- **OBS Studio** — Virtual camera driver ([download](https://obsproject.com/))
- **VB-Audio Virtual Cable** — Virtual mic device ([download](https://vb-audio.com/Cable/))

### Optional: OpenPose

> **Removed.** Model 2 previously used OpenPose; it is now **SignBart** running on
> MediaPipe Holistic. OpenPose binaries, the worker process, and the `tools/openpose`
> folder have been removed. No OpenPose installation is required.

### Required Models
- **Sign language model** (`.h5`, `.keras`, `.tflite`, or `.onnx`) in `src/models/`
- **Piper voice model** (`.onnx` + `.onnx.json`) in `src/models/` ([download](https://github.com/rhasspy/piper/releases))
- **Label map** (`labels.json` or `labels.txt`) in `src/models/` (optional — defaults to A-Z)

## Endpoints

| Endpoint | Type | Description |
|----------|------|-------------|
| `ws://127.0.0.1:8765/ws` | WebSocket | Main communication channel |
| `http://127.0.0.1:8765/health` | GET | Health check (JSON status) |
