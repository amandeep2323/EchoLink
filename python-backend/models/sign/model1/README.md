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
| Inference | Single-frame (no temporal window needed) |
| Landmarks | MediaPipe Hands (1 hand, 21 points) |
| Normalization | Min-max per axis (x, y independently normalized to [0,1]) |

## Files

```
model1/
├── model.json      ← Configuration (preprocessing, thresholds, metadata)
├── model.onnx      ← Model weights (converted from .h5 via Kaggle)
├── labels.json     ← Optional (labels also defined in model.json)
└── README.md       ← This file
```

## Setup

1. Convert your `.h5` model to `.onnx` using the Kaggle notebook (`kaggle_convert_to_onnx.py`)
2. Place `model.onnx` in this directory
3. Optionally place `labels.json` (the labels are also hardcoded in `model.json`)

## Model Details

- **Uses all 3 dimensions (x, y, z)** — the Conv1D first layer requires depth=3
- **Single-frame classification** — each frame is independently classified, no sliding window
- **Misrecognition fixes** — Geometric checks for commonly confused pairs: A/T, D/I, F/W
- **Spell correction** — completed words are spell-checked (e.g., "HELO" → "HELLO")

## Conversion

If you only have the `.h5` file:

```bash
# Use the Kaggle notebook
python kaggle_convert_to_onnx.py

# Or with the local converter (requires tensorflow + tf2onnx)
python convert_model.py model1/model.h5
```
