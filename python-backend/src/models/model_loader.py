"""
Model Loader — Unified Sign Language Model Interface
=======================================================
Auto-detects model format (.onnx, .h5, .keras, .tflite),
and provides a unified prediction API.

Config-driven loading (Phase 6):
    loader = ModelLoader()
    loader.load_from_config(config)   # Uses model.json config
    sign, confidence, top_3 = loader.predict_sign(features)

Legacy loading (backward compat):
    loader = ModelLoader()
    loader.load("path/to/model.onnx")
    loader.load_from_directory("models/sign/model1/")
"""

import os
import time
import numpy as np
from typing import Optional

from .label_map import LabelMap


# Backend type
BACKEND_ONNX = "onnx"
BACKEND_KERAS = "keras"


class ModelLoader:
    """
    Unified model loader with auto-detection and dual-backend inference.

    Supports:
      - .onnx   → Load directly with ONNX Runtime (recommended)
      - .h5     → Try cached .onnx → try convert → fallback to Keras
      - .keras  → Try cached .onnx → try convert → fallback to Keras
      - .tflite → Convert to ONNX

    Config-driven (Phase 6):
      - load_from_config(ModelConfig) — uses model.json for everything
      - Respects backend preference, labels, input shape from config
    """

    def __init__(self):
        # ONNX backend
        self._session = None           # onnxruntime.InferenceSession
        self._input_name: str = ""     # ONNX input tensor name
        self._output_name: str = ""    # ONNX output tensor name

        # Keras backend (fallback)
        self._keras_model = None       # keras.Model instance

        # Common
        self._input_shape: tuple = ()  # Expected input shape
        self._backend: str = ""        # "onnx" or "keras"
        self._labels: Optional[LabelMap] = None
        self._model_path: str = ""
        self._original_format: str = ""
        self._loaded: bool = False

        # Config-driven (Phase 6)
        self._config = None            # ModelConfig if loaded via config

    # ── Properties ──────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def input_shape(self) -> tuple:
        return self._input_shape

    @property
    def labels(self) -> Optional[LabelMap]:
        return self._labels

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def config(self):
        """Return the ModelConfig if loaded via config, else None."""
        return self._config

    @property
    def model_info(self) -> dict:
        info = {
            "loaded": self._loaded,
            "path": self._model_path,
            "original_format": self._original_format,
            "backend": self._backend,
            "input_shape": list(self._input_shape) if self._input_shape else [],
            "num_classes": self._labels.num_classes if self._labels else 0,
        }
        # Add config info if available
        if self._config:
            info["model_id"] = self._config.model_id
            info["model_name"] = self._config.name
            info["model_type"] = self._config.type
        return info

    # ── Config-Driven Loading (Phase 6) ─────────

    def load_from_config(self, config, use_gpu: bool = False) -> None:
        """
        Load a model using a ModelConfig from model.json.

        This is the primary loading method for Phase 6 multi-model support.
        Uses the config to determine:
          - Which file to load (config.model_path)
          - Which labels to use (config.labels_list)
          - Backend preference (config.inference.backend)
          - Input shape expectations (config.input.input_shape)
        """
        self._config = config
        model_path = config.model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Expected '{config.model_file}' in {config.model_dir}"
            )

        ext = os.path.splitext(model_path)[1].lower()
        self._original_format = ext.lstrip(".").upper()
        self._model_path = model_path

        print(f"[ModelLoader] Loading model: {config.name}")
        print(f"[ModelLoader]   File: {model_path} (format: {self._original_format})")
        print(f"[ModelLoader]   Type: {config.type}, Backend pref: {config.inference.backend}")

        start = time.time()

        # Determine loading strategy based on format and config preference
        preferred_backend = config.inference.backend  # "onnx" or "keras"

        if ext == ".onnx":
            # Direct ONNX — always use ONNX Runtime
            self._load_onnx_session(model_path, use_gpu)

        elif ext in (".h5", ".keras"):
            if preferred_backend == "keras":
                # User explicitly wants Keras
                print("[ModelLoader]   Config prefers Keras backend")
                self._load_keras_model(model_path)
            else:
                # Default: try ONNX first, fall back to Keras
                self._load_keras_with_fallback(model_path, use_gpu)

        elif ext == ".tflite":
            from .converter import ModelConverter
            onnx_path = ModelConverter.ensure_onnx(model_path)
            self._load_onnx_session(onnx_path, use_gpu)

        else:
            raise ValueError(f"Unsupported format: {ext}")

        # Set labels from config (no auto-discovery needed)
        labels_list = config.labels_list
        if labels_list:
            self._labels = LabelMap.from_list(labels_list)
            print(f"[ModelLoader]   Labels: {len(labels_list)} classes from config")
        else:
            # Fallback to file-based discovery
            self._labels = LabelMap.auto_discover(config.model_dir)

        # Validate label count matches model output
        self._validate_labels()

        elapsed = time.time() - start
        print(
            f"[ModelLoader] ✓ Model loaded in {elapsed:.2f}s "
            f"— backend: {self._backend}, input: {self._input_shape}, "
            f"classes: {self._labels.num_classes if self._labels else '?'}"
        )

    # ── Legacy Loading (backward compat) ────────

    def load(
        self,
        model_path: str,
        labels: Optional[LabelMap] = None,
        use_gpu: bool = False,
    ) -> None:
        """
        Load a model from any supported format (legacy method).

        For .onnx: loads directly with ONNX Runtime (no other deps needed).
        For .h5/.keras: tries cached ONNX first, then conversion, then Keras.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        ext = os.path.splitext(model_path)[1].lower()
        self._original_format = ext.lstrip(".").upper()
        self._model_path = model_path
        self._config = None  # Legacy mode — no config
        print(f"[ModelLoader] Loading model: {model_path} (format: {self._original_format})")

        start = time.time()

        if ext == ".onnx":
            self._load_onnx_session(model_path, use_gpu)
        elif ext in (".h5", ".keras"):
            self._load_keras_with_fallback(model_path, use_gpu)
        elif ext == ".tflite":
            from .converter import ModelConverter
            onnx_path = ModelConverter.ensure_onnx(model_path)
            self._load_onnx_session(onnx_path, use_gpu)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # Load labels
        if labels:
            self._labels = labels
        else:
            model_dir = os.path.dirname(os.path.abspath(model_path))
            self._labels = LabelMap.auto_discover(model_dir)

        self._validate_labels()

        elapsed = time.time() - start
        print(
            f"[ModelLoader] ✓ Model loaded in {elapsed:.2f}s "
            f"— backend: {self._backend}, input: {self._input_shape}, "
            f"classes: {self._labels.num_classes if self._labels else '?'}"
        )

    def load_from_directory(
        self,
        directory: str,
        use_gpu: bool = False,
    ) -> None:
        """Auto-discover and load a model from a directory (legacy)."""
        # Phase 6: check for model.json first
        config_path = os.path.join(directory, "model.json")
        if os.path.exists(config_path):
            from .model_config import ModelConfig
            config = ModelConfig.load(config_path)
            self.load_from_config(config, use_gpu=use_gpu)
            return

        # Legacy: scan for model files
        from .converter import ModelConverter
        model_path = ModelConverter.find_model_file(directory)
        if model_path is None:
            raise FileNotFoundError(
                f"No supported model file found in: {directory}\n"
                f"Supported formats: .onnx, .h5, .keras, .tflite\n"
                f"Place your model file in this directory."
            )
        self.load(model_path, use_gpu=use_gpu)

    # ── Keras with ONNX Fallback ────────────────

    def _load_keras_with_fallback(self, model_path: str, use_gpu: bool) -> None:
        """Try ONNX (cached or converted) first, fall back to Keras if needed."""

        # Check for cached ONNX file first
        onnx_path = os.path.splitext(model_path)[0] + ".onnx"
        if os.path.exists(onnx_path):
            if os.path.getmtime(onnx_path) >= os.path.getmtime(model_path):
                print(f"[ModelLoader] Found cached ONNX: {onnx_path}")
                try:
                    self._load_onnx_session(onnx_path, use_gpu)
                    return
                except Exception as e:
                    print(f"[ModelLoader] Cached ONNX failed: {e}")

        # Try converting to ONNX
        try:
            from .converter import ModelConverter
            onnx_path = ModelConverter.ensure_onnx(model_path)
            self._load_onnx_session(onnx_path, use_gpu)
            return
        except Exception as e:
            print(f"[ModelLoader] ⚠ ONNX conversion failed: {e}")
            print(f"[ModelLoader] Falling back to Keras direct inference...")

        # Fallback: load with Keras directly
        self._load_keras_model(model_path)

    # ── ONNX Session ───────────────────────────

    def _load_onnx_session(self, onnx_path: str, use_gpu: bool = False) -> None:
        """Initialize the ONNX Runtime inference session."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                f"ONNX Runtime not installed: {e}\n"
                f"Install with: pip install onnxruntime"
            ) from e

        providers = []
        if use_gpu:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
                print("[ModelLoader] Using CUDA GPU acceleration")
            else:
                print("[ModelLoader] CUDA not available — using CPU")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
        )

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()

        if not inputs or not outputs:
            raise RuntimeError("ONNX model has no inputs or outputs")

        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._backend = BACKEND_ONNX
        self._loaded = True

        # Parse input shape — handle dynamic dims
        raw_shape = inputs[0].shape
        parsed_shape = []
        for dim in raw_shape:
            if isinstance(dim, int):
                parsed_shape.append(dim)
            else:
                parsed_shape.append(None)  # dynamic dimension
        self._input_shape = tuple(parsed_shape)

        print(
            f"[ModelLoader] ONNX session — "
            f"provider: {self._session.get_providers()[0]}, "
            f"input: {self._input_name} {self._input_shape}, "
            f"output: {self._output_name}"
        )

    # ── Keras Direct Loading ────────────────────

    def _load_keras_model(self, model_path: str) -> None:
        """Load model directly with Keras for inference."""
        from .converter import ModelConverter
        self._keras_model = ModelConverter.load_keras_model(model_path)

        self._input_shape = tuple(self._keras_model.input_shape)
        self._backend = BACKEND_KERAS
        self._loaded = True

        print(
            f"[ModelLoader] Keras model — "
            f"input: {self._input_shape}, "
            f"output: {self._keras_model.output_shape}"
        )

    # ── Validation ──────────────────────────────

    def _validate_labels(self) -> None:
        """Warn if label count doesn't match model output dimension."""
        if not self._labels:
            return

        num_output = None

        if self._backend == BACKEND_ONNX and self._session:
            outputs = self._session.get_outputs()
            if outputs and outputs[0].shape:
                shape = outputs[0].shape
                if len(shape) >= 1 and isinstance(shape[-1], int):
                    num_output = shape[-1]

        elif self._backend == BACKEND_KERAS and self._keras_model:
            output_shape = self._keras_model.output_shape
            if output_shape and len(output_shape) >= 1:
                num_output = output_shape[-1]

        if num_output is not None and num_output != self._labels.num_classes:
            print(
                f"[ModelLoader] ⚠ Label mismatch: model outputs "
                f"{num_output} classes but label map has "
                f"{self._labels.num_classes} labels"
            )

    # ── Prediction ──────────────────────────────

    def predict_raw(self, features: np.ndarray) -> np.ndarray:
        """
        Run raw inference. Returns the model's output tensor.
        Works with both ONNX and Keras backends.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded — call load() first")

        if features.dtype != np.float32:
            features = features.astype(np.float32)

        if self._backend == BACKEND_ONNX:
            outputs = self._session.run(
                [self._output_name],
                {self._input_name: features},
            )
            return outputs[0]

        elif self._backend == BACKEND_KERAS:
            predictions = self._keras_model.predict(features, verbose=0)
            return predictions

        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference and return sorted class probabilities.
        Returns (indices, probabilities) sorted by confidence descending.
        """
        raw_output = self.predict_raw(features)

        probs = raw_output
        if probs.ndim > 1:
            probs = probs[0]

        # Apply softmax if outputs aren't probabilities
        should_softmax = True
        if self._config and not self._config.inference.apply_softmax:
            should_softmax = False

        if should_softmax:
            if probs.min() < 0 or probs.max() > 1.0 or abs(probs.sum() - 1.0) > 0.1:
                probs = _softmax(probs)

        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        return sorted_indices, sorted_probs

    def predict_sign(
        self,
        features: np.ndarray,
        top_k: int = 3,
    ) -> tuple[str, float, list[dict]]:
        """
        Predict the sign from features and return human-readable results.

        Returns:
            (sign_name, confidence, top_k_list)
        """
        indices, probs = self.predict(features)

        labels = self._labels or LabelMap.default()

        best_sign = labels.get_label(int(indices[0]))
        best_conf = float(probs[0])

        top_k_list = []
        for i in range(min(top_k, len(indices))):
            top_k_list.append({
                "sign": labels.get_label(int(indices[i])),
                "confidence": float(probs[i]),
            })

        return best_sign, best_conf, top_k_list

    # ── Cleanup ─────────────────────────────────

    def unload(self) -> None:
        """Release the model and free memory."""
        if self._session:
            del self._session
            self._session = None
        if self._keras_model:
            del self._keras_model
            self._keras_model = None
        self._loaded = False
        self._backend = ""
        self._input_name = ""
        self._input_shape = ()
        self._output_name = ""
        self._labels = None
        self._model_path = ""
        self._config = None
        print("[ModelLoader] Model unloaded")

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
