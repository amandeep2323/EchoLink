# SignSpeak — ASL to Speech Desktop Application

SignSpeak is a desktop application that captures your webcam, detects American Sign Language gestures in real time using machine learning, converts detected signs into a text transcript, overlays that transcript onto the camera feed, outputs the composited video through a virtual camera, and speaks the transcript via TTS through a virtual microphone.

## Quick Start (Windows)

Double-click `start.bat` in the project root. This opens two PowerShell windows:

1. **Backend** — Activates Python venv and starts WebSocket server on `ws://127.0.0.1:8765/ws`
2. **Frontend** — Installs npm dependencies if needed and starts Vite dev server on `http://localhost:5173`

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  RENDERER (React + TypeScript + Tailwind CSS)                   │
│  ├── Camera preview (base64 JPEG frames from Python backend)   │
│  ├── Live transcript panel (scrolling text, copy, export)      │
│  ├── Sign detection display (sign + confidence + top 3)        │
│  ├── Control bar (Start/Stop, Landmarks, Overlay, TTS, VCam,  │
│  │               VMic toggles, threshold slider, camera select)│
│  ├── Toast notification system (pipeline events, errors)       │
│  ├── Settings panel (resolution, FPS, voice, device selection) │
│  ├── Session stats (duration, word count, frame count)         │
│  └── Status indicators (pipeline, model, hands, VCam, VMic)   │
└──────────────────────┬──────────────────────────────────────────┘
                       │ WebSocket (ws://127.0.0.1:8765/ws)
┌──────────────────────┴──────────────────────────────────────────┐
│  PYTHON BACKEND (FastAPI WebSocket Server on port 8765)         │
│  ├── Camera Capture (OpenCV, threaded, disconnect recovery)    │
│  ├── Landmark Extraction (MediaPipe Hands, 21 landmarks)       │
│  ├── Sign Classifier (PointNet → ONNX Runtime, 24 letters)    │
│  ├── Letter Accumulation (smoothing, stability, spell correct) │
│  ├── Frame Compositor (transcript bar, sign box, hand label)   │
│  ├── Virtual Camera (pyvirtualcam → OBS Virtual Camera)        │
│  ├── TTS Engine (pyttsx3 Windows SAPI / Piper fallback)        │
│  ├── Virtual Mic (sounddevice → VB-Audio Virtual Cable)        │
│  └── Health Check (GET /health endpoint)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Start / Stop pipeline |
| `Ctrl+,` | Open / Close settings |
| `Ctrl+L` | Toggle landmarks |
| `Ctrl+T` | Toggle TTS |
| `Ctrl+M` | Toggle VMic |
| `Ctrl+E` | Export transcript as .txt |
| `Ctrl+Shift+C` | Clear transcript |
| `Escape` | Close settings / modals |

## Setup

### 1. Python Backend

```bash
cd python-backend
python -m venv venv

# Activate virtual environment
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux/macOS

pip install -r requirements.txt
```

### 2. Place Model Files

```
python-backend/models/
├── sign/                          ← Sign language model
│   ├── model.onnx                 ← Your ONNX model (from Kaggle conversion)
│   └── labels.json                ← Label map (from Kaggle conversion)
│
└── tts/                           ← TTS voice (optional — pyttsx3 works without this)
    ├── en_US-lessac-medium.onnx         ← Piper voice model
    └── en_US-lessac-medium.onnx.json    ← Piper voice config
```

**Sign model formats**: `.onnx` (recommended), `.h5`, `.keras`, `.tflite` (auto-detected)

**Alphabet JSONs**: NOT NEEDED — the 24-letter label map is hardcoded.

### 3. Frontend

```bash
# From project root
npm install
```

### 4. Required Software (for VCam/VMic)

- **OBS Studio** — [download](https://obsproject.com/) — install and run once to register virtual camera driver
- **VB-Audio Virtual Cable** — [download](https://vb-audio.com/Cable/) — install for virtual microphone

## Running

### Option 1: One-Click (Windows)

```bash
# Double-click start.bat or run:
start.bat
```

### Option 2: Manual

```bash
# Terminal 1: Backend
cd python-backend
venv\Scripts\activate
python main.py

# Terminal 2: Frontend
npm run dev
```

### Endpoints

| Endpoint | URL |
|----------|-----|
| Frontend | http://localhost:5173 |
| WebSocket | ws://127.0.0.1:8765/ws |
| Health Check | http://127.0.0.1:8765/health |

## Features

### Phase 5c: Robustness & Quality of Life

| Feature | Description |
|---------|-------------|
| **start.bat** | One-click launcher: finds Python venv, starts backend + frontend |
| **Camera Recovery** | Auto-recovers from camera disconnect (up to 3 attempts, 3s apart) |
| **Auto-Restart** | Resolution/FPS/camera changes restart camera without stopping pipeline |
| **Duplicate Start Prevention** | Can't click Start twice during initialization |
| **Export Transcript** | Save transcript as `.txt` file with metadata (Ctrl+E) |
| **Session Stats** | Live duration, word count, frame count in footer |
| **Graceful Shutdown** | First Ctrl+C = graceful, second = force quit |
| **New Shortcuts** | Ctrl+M (VMic), Ctrl+E (Export), Escape (close settings) |

### Using with Google Meet / Zoom / Teams

1. Start the pipeline (click **▶ Start** or press `Ctrl+Enter`)
2. Enable **📷 VCam** → select **OBS Virtual Camera** in your video app
3. Enable **🎤 VMic** → select **VB-Audio Virtual Cable** as your microphone
4. Enable **🔊 TTS** to speak completed sentences
5. Participants see the transcript overlay and hear TTS audio

## WebSocket Protocol

### Client → Server

| Message Type | Data | Description |
|---|---|---|
| `start_pipeline` | `PipelineSettings` | Start the recognition pipeline |
| `stop_pipeline` | `null` | Stop the pipeline |
| `update_settings` | `Partial<PipelineSettings>` | Update settings live |
| `get_devices` | `null` | Request available devices |
| `clear_transcript` | `null` | Clear transcript and reset recognition |

### Server → Client

| Message Type | Data | Description |
|---|---|---|
| `preview_frame` | `{ frame: string }` | Base64 JPEG preview frame |
| `sign_detected` | `{ sign, confidence, smoothed_confidence, letter_added, top_3 }` | Sign detection |
| `transcript_update` | `{ full_text, latest_word, is_sentence_complete }` | Transcript change |
| `status_update` | `StatusData` | Pipeline status |
| `device_list` | `{ cameras, audio_output_devices }` | Available devices |
| `error` | `{ message: string }` | Error notification |

## Error Handling

### Frontend
- **Silent health polling** — checks backend every 3s, auto-connects when available
- **Toast notifications** — pipeline events shown as non-blocking toasts
- **Error banners** — backend errors shown as dismissable banners (auto-dismiss 8s)
- **Stale connection detection** — warns if no messages received for 30s

### Backend
- **Camera recovery** — auto-recovers from disconnect (3 attempts, 3s apart)
- **Auto-restart on settings change** — resolution/FPS/camera changes restart camera
- **Duplicate start prevention** — pipeline can't be started twice simultaneously
- **Auto-stop on errors** — stops after 10 consecutive processing errors
- **Graceful shutdown** — SIGINT: clean stop; double SIGINT: force quit
- **Health endpoint** — `GET /health` for monitoring

## Diagnostics

```bash
cd python-backend
python diagnose.py
```

Tests: Python version, dependencies, MediaPipe, ONNX Runtime, model loading, TTS synthesis.
