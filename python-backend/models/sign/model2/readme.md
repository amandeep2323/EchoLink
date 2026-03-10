# Model 2 — WLASL Pose-TGCN (2000 Words)

## Overview

| Property | Value |
|----------|-------|
| Name | WLASL Pose-TGCN (2000 Words) |
| Architecture | Multi-Attention Spatial-Temporal Graph Convolutional Network |
| Source | [dxli94/WLASL](https://github.com/dxli94/WLASL) |
| Input | `(1, 55, 100)` — 55 keypoints × 50 frames × 2 (x,y) |
| Output | `(1, 2000)` — 2000 ASL words |
| Inference | `sequence` — 50-frame rolling buffer |
| Landmarks | `mediapipe_holistic` — 55 upper-body keypoints |
| Normalization | `none` — raw coordinates |

## Files

```
model2/
├── model.json              ← Configuration (pipeline settings, thresholds)
├── wlasl_pose_tgcn.onnx    ← Model weights (converted from PyTorch)
├── labels.json             ← 2000-word label map (integer → English word)
└── readme.md               ← This file
```

## Pipeline Behavior

When this model is active, `model.json` drives the pipeline to:
- Use **MediaPipe Holistic** for landmark extraction (55 points)
- Extract **13 upper-body/face** + **21 left-hand** + **21 right-hand** keypoints per frame
- Buffer **50 frames** into a rolling `deque`
- Format the buffer into a tensor of shape `[1, 55, 100]` (55 nodes × 50 frames × 2 coords)
- Classify the full sequence via ONNX Runtime
- Apply **confidence smoothing** (no letter accumulation or spell correction)
- Emit the predicted **full word** directly when smoothed confidence exceeds threshold
- **Clear the buffer** after emission to prevent stutter-repeating

## 55-Point Keypoint Layout

| Index Range | Count | Source | Description |
|-------------|-------|--------|-------------|
| 0–10 | 11 | Pose landmarks | Nose, shoulders, elbows, wrists, eyes, ears |
| 11 | 1 | Computed | Neck (midpoint of shoulders) |
| 12 | 1 | Computed | Mid-hip (midpoint of hips) |
| 13–33 | 21 | Left hand | All 21 MediaPipe hand landmarks |
| 34–54 | 21 | Right hand | All 21 MediaPipe hand landmarks |

Hands are zero-padded when not visible.

## Credits

- **Original Paper**: *Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison* (WACV 2020)
- **Authors**: Dongxu Li, Cristian Rodriguez, Xin Yu, Hongdong Li
- **Repository**: [dxli94/WLASL](https://github.com/dxli94/WLASL)
