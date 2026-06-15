# Model 4 — ASL-Citizen GRU (2731 Words)

Word/gloss-level ASL recognition using a GRU sequence model trained on the
[ASL-Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/)
dataset. Recognizes **2731 gloss classes**.

## Architecture

`Sequential( GRU(512, return_sequences=True) → Dropout(0.4) → GRU(256) → Dropout(0.4) → Dense(2731, softmax) )`

- Input: `(1, 150, 172)` float32 — 150 timesteps × 172 features
- Output: `(1, 2731)` softmax probabilities (softmax already applied by the model)
- Params: ~7.0M

## Input features (172 = 86 landmarks × 2)

MediaPipe Holistic keypoints (x, y only), per frame, in this exact order:

| Block | Count | Array indices |
|-------|-------|---------------|
| Pose (filtered: drop face-region 0–10 and legs 23–32) | 12 | 0–11 |
| Left hand | 21 | 12–32 |
| Right hand | 21 | 33–53 |
| Face (32 selected landmarks) | 32 | 54–85 |

Each frame is **anchor-normalized** (neck-relative global norm, face anchor at
index 79, left/right arm norms, and per-hand bounding-box normalization), then a
clip is **tiled/padded to exactly 150 frames** and reshaped to `(150, 172)`.

## Runtime

- **Backend: OpenVINO Runtime** using the IR `model4.xml` / `model4.bin`.
- The GRU's ONNX export emits a `Loop` operator OpenVINO cannot compile (same
  issue as Model 3). This was avoided by converting through the TensorFlow
  SavedModel frontend, which maps the GRU to native recurrent ops.
- `model.onnx` is kept only as a diagnostic / runtime fallback.

## Files

- `model4.xml` / `model4.bin` — OpenVINO IR (inference artifact)
- `labels.json` — 2731 index→gloss map (normalized)
- `model.json` — pipeline config
- `best_model_2731.keras`, `index_to_gloss_2731.json` — original sources
- `model.onnx` — fallback/diagnostic
- `export_savedmodel.py` — Keras → SavedModel (run in `model4_tf_venv`, TensorFlow)
- `savedmodel_to_ir.py` — SavedModel → OpenVINO IR (run in `.venv`, OpenVINO)
- `convert_keras_to_onnx.py` — optional ONNX export (run in `model4_tf_venv`)
- `test_model4_local.py`, `test_vidoes/` — offline verification (filename = ground truth)

## Reproduce the conversion

```bat
REM 1) Export SavedModel (TensorFlow environment)
..\..\..\..\model4_tf_venv\Scripts\python.exe export_savedmodel.py

REM 2) Convert to OpenVINO IR (app .venv with OpenVINO)
..\..\..\..\.venv\Scripts\python.exe savedmodel_to_ir.py

REM 3) Verify on sample clips
..\..\..\..\.venv\Scripts\python.exe test_model4_local.py
```

## Verified accuracy

9/10 on the bundled `test_vidoes/` clips; OpenVINO IR predictions are identical
to the original Keras model.
