# Model 1 — PointNet ASL Fingerspelling

## Overview

| Property | Value |
|----------|-------|
| Name | PointNet ASL Fingerspelling |
| Architecture | PointNet (point cloud classifier with Conv1D layers) |
| Source | [kevinjosethomas/sign-language-processing](https://github.com/kevinjosethomas/sign-language-processing) |
| Input | `(1, 21, 3)` — 21 hand landmarks × (x, y, z) |
| Output | `(1, 24)` — 24 ASL letters |
| Letters | `ABCDEFGHIKLMNOPQRSTUVWXY` (J, Z excluded — require motion) |
| Inference | `single_frame` — no temporal window needed |
| Landmarks | `mediapipe_hands` — 1 hand, 21 points |
| Normalization | Min-max per axis (x, y independently normalized to [0,1]) |

## Files

```
model1/
├── model.json      ← Configuration (preprocessing, thresholds, metadata)
├── model.onnx      ← Model weights (converted from .h5 via Kaggle)
├── labels.json     ← Optional (labels also defined in model.json)
└── README.md       ← This file
```

## Pipeline Behavior

When this model is active, `model.json` drives the pipeline to:
- Use **MediaPipe Hands** for landmark extraction (21 points)
- Classify each **single frame** independently via ONNX Runtime
- Apply **misrecognition fixes** (geometric checks for A/T, D/I, F/W)
- **Accumulate letters** into words with stability gates and cooldowns
- Run **spell correction** when words are finalized

## Conversion

If you only have the `.h5` file:

```bash
# Use the Kaggle notebook
python kaggle_convert_to_onnx.py

# Or with the local converter (requires tensorflow + tf2onnx)
python convert_model.py model1/model.h5
```
