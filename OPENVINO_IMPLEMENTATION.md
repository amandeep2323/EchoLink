# Intel OpenVINO Model Acceleration — Implementation Guide

## Overview

EchoLink's sign language recognition runs through a **hybrid inference architecture**:

| Model | Architecture | Backend | Reason |
|-------|--------------|---------|--------|
| Model 1 | PointNet (fingerspelling, 24 classes) | **Intel OpenVINO** | Fully compatible, Intel CPU optimizations |
| Model 2 | WLASL Pose-TGCN (2000 words) | **Intel OpenVINO** | Fully compatible, Intel CPU optimizations |
| Model 3 | LSTM (250 signs) | **ONNX Runtime** | Contains `Loop` operator unsupported by OpenVINO |

The system automatically selects the optimal runtime per model. No frontend, WebSocket, pipeline, or API changes were required — integration is fully transparent.

## Architecture

All inference flows through a single class: `python-backend/src/models/model_loader.py` → `ModelLoader`.

```
load_from_config(config)
        │
        ├── .onnx file?
        │     ├── _is_model_openvino_compatible() == True
        │     │        └── _load_openvino_model()   → backend = "openvino"
        │     └── _is_model_openvino_compatible() == False (Model 3)
        │              └── _load_onnx_session()      → backend = "onnx"
        │
        └── .h5 / .keras → _load_keras_with_fallback()

predict_raw(features)
        ├── backend == "openvino" → infer_request.infer(...)
        ├── backend == "onnx"     → session.run(...)
        └── backend == "keras"    → keras_model.predict(...)
```

`predict_raw()` returns an identical `np.ndarray` regardless of backend, so all
downstream code (`predict()`, `predict_sign()`, the pipeline manager, WebSocket
responses) is unchanged.

### Smart Backend Detection

`_is_model_openvino_compatible()` returns `False` for any path containing
`model3` (the LSTM model with the unsupported `Loop` operator) and `True`
otherwise. If OpenVINO loading fails for any reason, `load_from_config()` falls
back to ONNX Runtime automatically, so the system degrades gracefully.

## Performance Optimization

Models are compiled with the **LATENCY** performance hint, which optimizes for
single-frame real-time recognition rather than batch throughput:

```python
config = {"PERFORMANCE_HINT": "LATENCY"}
compiled = core.compile_model(model=model, device_name=device, config=config)
```

- **Device**: defaults to `CPU`. Passing `use_gpu=True` requests `AUTO`, which
  lets OpenVINO pick the best available device and falls back to `CPU` if the
  preferred device is unavailable.
- **Single inference request** per loaded model, reused across frames.

## Model Caching

OpenVINO caches compiled models to disk so subsequent launches skip recompilation.

- **Cache directory**: `cache/openvino/` (auto-created on first run).
- **OpenVINO blob cache**: managed by OpenVINO via `core.set_property({"CACHE_DIR": ...})`.
- **Metadata sidecar**: `cache/openvino/{cache_key}/cache.meta` — a JSON file
  written by `_save_cache_metadata()` recording source path, source mtime,
  OpenVINO version, device, and performance hint.

### Cache key

`_generate_cache_key()` produces a 16-char MD5 hash of:

```
{absolute_model_path}_{device}_LATENCY_{openvino_version}
```

This guarantees a fresh compile when the model path, device, performance hint,
or OpenVINO version changes.

### Cache validation

`_check_cache_status()` returns one of:

- `Cache hit` — valid cache present, fast warm start.
- `Cache miss` — no cache directory/metadata.
- `Cache miss (source modified)` — the ONNX file is newer than the cached copy.
- `Cache miss (validation failed)` — metadata unreadable.

### Cache logs

```
[OpenVINO] Cache miss      ← first run, compiles and stores
[OpenVINO] Cache hit       ← subsequent runs, loads from cache
```

Observed warm-start improvements during testing ranged from ~25% to ~87% faster
load times depending on the model.

## Logging

Startup and per-model logs make backend selection observable:

```
[OpenVINO] Runtime initialized
[OpenVINO] Device: CPU
[OpenVINO] Model Cache Enabled: cache/openvino/
[OpenVINO] Loading model: .../model1/model.onnx
[OpenVINO] Cache hit
[OpenVINO] Performance Hint: LATENCY
[OpenVINO] Model compiled with LATENCY hint
[OpenVINO] Loaded Model 1
```

Model 3 logs the ONNX Runtime path:

```
[ModelLoader] Model not compatible with OpenVINO, using ONNX Runtime
[ONNX Runtime] Loading model: .../model3/model.onnx
[ONNX Runtime] Execution providers: ['CPUExecutionProvider']
[ONNX Runtime] Model loaded successfully
```

## Dependencies

`python-backend/requirements.txt`:

```
openvino>=2024.0.0      # Primary engine for Models 1 & 2
onnxruntime>=1.22.0     # Fallback engine for Model 3
```

Both runtimes are required for the hybrid architecture. No TensorFlow or PyTorch
inference dependency is introduced, and the original `.onnx` model files are used
unchanged (no IR conversion, no retraining).

## Troubleshooting

**`Intel OpenVINO Runtime is not installed`**
Install with `pip install openvino>=2024.0.0`. Models 1 & 2 require it.

**`Model contains unsupported operators`**
The model has an operator OpenVINO's ONNX frontend can't convert (e.g. `Loop`).
Add the model's directory name to the `model3`-style exclusion in
`_is_model_openvino_compatible()` so it routes to ONNX Runtime.

**`Input shape mismatch for Model X`**
The preprocessing pipeline produced the wrong tensor shape. The error message
prints the expected shape (with `None` marking dynamic dimensions) and the shape
received. Verify the feature extractor output.

**Stale predictions after replacing an `.onnx` file**
The cache invalidates automatically when the source file mtime changes. To force
a clean rebuild, delete the `cache/openvino/` directory.

**`AUTO` device fails**
The loader logs a warning and retries on `CPU` automatically. To force CPU, load
with `use_gpu=False` (the default).

## Migration Notes for Developers

- **No API changes.** `ModelLoader.load_from_config()`, `predict_raw()`,
  `predict()`, and `predict_sign()` keep identical signatures and return types.
- **Backend is transparent.** Check `loader.backend` (`"openvino"`, `"onnx"`, or
  `"keras"`) if you need to know which engine handled a model.
- **Adding a new OpenVINO-incompatible model.** Extend
  `_is_model_openvino_compatible()` to return `False` for it; it will use ONNX
  Runtime automatically.
- **Cache location.** Configurable via `ModelLoader._cache_dir`; caching can be
  disabled by setting `ModelLoader._cache_enabled = False`.
