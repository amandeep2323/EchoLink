# Sign Language Models Directory

## Structure

Each model lives in its own subfolder with a `model.json` config file:

```
models/sign/
├── model1/                    ← PointNet Fingerspelling (default)
│   ├── model.json             ← REQUIRED — model configuration
│   ├── model.onnx             ← Model weights
│   ├── labels.json            ← Optional label map
│   └── README.md              ← Model documentation
│
├── model2/                    ← Your next model
│   ├── model.json
│   ├── your_model.onnx
│   └── README.md
│
└── _active_model.txt          ← Auto-generated, persists selection
```

## Adding a New Model

1. Create a new folder: `models/sign/model2/`
2. Place your model weights file (`.onnx`, `.h5`, `.keras`, or `.tflite`)
3. Create a `model.json` — copy from `model1/model.json` and modify:

```json
{
  "name": "Your Model Name",
  "model_file": "your_model.onnx",
  "version": "1.0",
  "author": "Your Name",
  "description": "What this model does",
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
    "spell_correction": true,
    "stability_frames": 5,
    "repeat_frames": 15
  }
}
```

4. Restart the backend — the model will be auto-discovered
5. Switch to it in Settings → Model Selection

## Required Fields

Only `name` and `model_file` are required. Everything else has defaults
that match the current PointNet model.

## Supported Model Types

| Type | Input | Inference | Example |
|------|-------|-----------|---------|
| `fingerspelling` | Single hand landmarks | Single frame | Current PointNet model |
| `word_recognition` | Hand + pose landmarks | Frame sequence (LSTM) | Future models |

## Legacy Support

If you have model files directly in `models/sign/` (not in a subfolder),
the registry will auto-migrate them to `models/sign/model1/` on first run.
