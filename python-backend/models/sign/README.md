# Sign Language Models Directory

## Structure

Each model lives in its own subfolder with a `model.json` config file:

```
models/sign/
├── model1/                    ← PointNet Fingerspelling (default)
│   ├── model.json             ← REQUIRED — model configuration
│   ├── model.onnx             ← Model weights (24-letter classifier)
│   ├── labels.json            ← Optional label map
│   └── README.md              ← Model documentation
│
├── model2/                    ← WLASL Pose-TGCN (2000 words)
│   ├── model.json             ← REQUIRED — model configuration
│   ├── wlasl_pose_tgcn.onnx   ← Model weights (word-level classifier)
│   ├── labels.json            ← 2000-word label map
│   └── readme.md              ← Model documentation
│
└── _active_model.txt          ← Auto-generated, persists selection
```

## Adding a New Model

1. Create a new folder: `models/sign/model3/`
2. Place your model weights file (`.onnx`, `.h5`, `.keras`, or `.tflite`)
3. Create a `model.json` — use one of the templates below:

### Template: Single-Frame Fingerspelling

```json
{
  "name": "Your Model Name",
  "model_file": "your_model.onnx",
  "type": "fingerspelling",
  "labels": "ABCDEFGHIKLMNOPQRSTUVWXY",
  "input": {
    "landmark_source": "mediapipe_hands",
    "max_hands": 1,
    "input_shape": [1, 21, 3],
    "use_dimensions": "auto",
    "normalize": "min_max"
  },
  "inference": {
    "type": "single_frame",
    "confidence_threshold": 0.60,
    "backend": "onnx"
  },
  "postprocess": {
    "misrecognition_fixes": true,
    "spell_correction": true
  }
}
```

### Template: Sequence Word-Level (Holistic)

```json
{
  "name": "Your Word Model",
  "model_file": "your_model.onnx",
  "type": "word_level",
  "labels": "labels.json",
  "input": {
    "landmark_source": "mediapipe_holistic",
    "max_hands": 2,
    "input_shape": [1, 55, 100],
    "use_dimensions": 2,
    "normalize": "none"
  },
  "inference": {
    "type": "sequence",
    "confidence_threshold": 0.30,
    "sequence_length": 50,
    "backend": "onnx"
  },
  "postprocess": {
    "misrecognition_fixes": false,
    "spell_correction": false
  }
}
```

4. Restart the backend — the model will be auto-discovered
5. Switch to it in Settings → Model Selection

## Required Fields

Only `name` and `model_file` are required. Everything else has defaults
that match the current PointNet model.

## Supported Model Types

| Type | Landmark Source | Inference | Points | Example |
|------|----------------|-----------|--------|---------|
| `fingerspelling` | `mediapipe_hands` | `single_frame` | 21 hand | Model 1 — PointNet |
| `word_level` | `mediapipe_holistic` | `sequence` | 55 upper-body | Model 2 — WLASL Pose-TGCN |

## Key Configuration Fields

| Field | Purpose | Options |
|-------|---------|---------|
| `input.landmark_source` | Which MediaPipe model to use | `mediapipe_hands`, `mediapipe_holistic` |
| `inference.type` | Frame vs. temporal classification | `single_frame`, `sequence` |
| `inference.sequence_length` | Frames to buffer (sequence only) | Integer (e.g., 50) |
| `input.normalize` | Landmark normalization method | `min_max`, `wrist_relative`, `none` |

## Legacy Support

If you have model files directly in `models/sign/` (not in a subfolder),
the registry will auto-migrate them to `models/sign/model1/` on first run.
