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
BACKEND_SIGNBART = "signbart"


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
        # OpenVINO backend
        self._core = None              # openvino.runtime.Core instance
        self._compiled_model = None    # OpenVINO CompiledModel
        self._infer_request = None     # OpenVINO InferRequest
        self._device: str = "CPU"      # OpenVINO device ("CPU" or "AUTO")
        
        # ONNX backend (legacy)
        self._session = None           # onnxruntime.InferenceSession
        self._input_name: str = ""     # ONNX input tensor name
        self._output_name: str = ""    # ONNX output tensor name

        # Keras backend (fallback)
        self._keras_model = None       # keras.Model instance

        # Common
        self._input_shape: tuple = ()  # Expected input shape
        self._backend: str = ""        # "onnx", "openvino", or "keras"
        self._labels: Optional[LabelMap] = None
        self._model_path: str = ""
        self._original_format: str = ""
        self._loaded: bool = False

        # Caching (OpenVINO)
        self._cache_dir: str = "cache/openvino/"
        self._cache_enabled: bool = True

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
        preferred_backend = config.inference.backend  # "onnx", "keras", "signbart"

        # SignBart (dual-input, OpenVINO Runtime) — handle before format dispatch
        # since its model_file may be an .xml IR.
        if config.inference.backend == "signbart":
            self._load_signbart(model_path)

        elif config.inference.backend == "openvino" or ext == ".xml":
            # Generic single-input OpenVINO IR (.xml) — e.g. Model 4 (GRU).
            # Loaded directly through the OpenVINO Runtime with LATENCY hint
            # and model cache (handled inside _load_openvino_model).
            try:
                device = "AUTO" if use_gpu else "CPU"
                self._load_openvino_model(model_path, device)
            except Exception as e:
                # Runtime fallback: try a sibling model.onnx via ONNX Runtime.
                onnx_fallback = os.path.join(config.model_dir, "model.onnx")
                print(f"[ModelLoader] OpenVINO IR load failed: {e}")
                if os.path.exists(onnx_fallback):
                    print(f"[ModelLoader] Falling back to ONNX Runtime: {onnx_fallback}")
                    self._load_onnx_session(onnx_fallback, use_gpu)
                else:
                    raise

        elif ext == ".onnx":
            # Check OpenVINO compatibility first
            if self._is_model_openvino_compatible(model_path):
                # Prefer a pre-converted OpenVINO IR (.xml) if it exists next to
                # the ONNX file — produced by models/sign/convert_models_to_ir.py.
                ir_path = os.path.splitext(model_path)[0] + ".xml"
                load_target = ir_path if os.path.exists(ir_path) else model_path
                if load_target == ir_path:
                    print(f"[ModelLoader]   Found OpenVINO IR: {os.path.basename(ir_path)}")
                try:
                    device = "AUTO" if use_gpu else "CPU"
                    self._load_openvino_model(load_target, device)
                except Exception as e:
                    print(f"[ModelLoader] OpenVINO failed: {e}")
                    print(f"[ModelLoader] Falling back to ONNX Runtime...")
                    self._load_onnx_session(model_path, use_gpu)
            else:
                # Model 3 or other incompatible models — use ONNX Runtime
                print(f"[ModelLoader] Model not compatible with OpenVINO, using ONNX Runtime")
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

    # ── OpenVINO Backend ────────────────────────

    def _initialize_openvino_core(self, device: str = "CPU") -> None:
        """
        Initialize OpenVINO Core with configuration.
        
        Args:
            device: Target device ("CPU" or "AUTO")
        
        Logs:
            "[OpenVINO] Runtime initialized"
            "[OpenVINO] Device: {device}"
            "[OpenVINO] Model Cache Enabled: {cache_dir}"
        """
        try:
            from openvino import Core
        except ImportError as e:
            raise RuntimeError(
                "Intel OpenVINO Runtime is not installed.\n\n"
                "To install OpenVINO:\n"
                "  pip install openvino>=2024.0.0\n\n"
                "For more information:\n"
                "  https://docs.openvino.ai/latest/get_started.html"
            ) from e
        
        self._core = Core()
        self._device = device
        
        # Set cache directory
        if self._cache_enabled:
            os.makedirs(self._cache_dir, exist_ok=True)
            self._core.set_property({"CACHE_DIR": self._cache_dir})
        
        print(f"[OpenVINO] Runtime initialized")
        print(f"[OpenVINO] Device: {device}")
        if self._cache_enabled:
            print(f"[OpenVINO] Model Cache Enabled: {self._cache_dir}")

    def _load_openvino_model(self, onnx_path: str, device: str = "CPU") -> None:
        """
        Load and compile ONNX model with OpenVINO.
        
        Args:
            onnx_path: Path to ONNX model file
            device: Target device ("CPU" or "AUTO")
        
        Raises:
            RuntimeError: If OpenVINO is not installed
            ValueError: If model has unsupported operators
            FileNotFoundError: If ONNX file doesn't exist
        
        Logs:
            "[OpenVINO] Loading model: {path}"
            "[OpenVINO] Performance Hint: LATENCY"
            "[OpenVINO] Model compiled with LATENCY hint"
            "[OpenVINO] Cache hit" or "[OpenVINO] Cache miss"
            "[OpenVINO] Loaded Model {1|2|3}"
        """
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        # Initialize core if needed
        if self._core is None:
            self._initialize_openvino_core(device)
        
        print(f"[OpenVINO] Loading model: {onnx_path}")
        
        try:
            # Read ONNX model
            model = self._core.read_model(model=onnx_path)
        except Exception as e:
            if "not supported" in str(e).lower():
                raise ValueError(
                    f"Model contains unsupported operators: {onnx_path}\n\n"
                    f"OpenVINO Error: {e}\n\n"
                    f"This model cannot be loaded with OpenVINO Runtime.\n"
                    f"Please check model compatibility:\n"
                    f"  https://docs.openvino.ai/latest/openvino_docs_MO_DG_"
                    f"prepare_model_convert_model_Convert_Model_From_ONNX.html"
                ) from e
            raise
        
        # Get input/output names
        self._input_name = model.input(0).get_any_name()
        self._output_name = model.output(0).get_any_name()
        
        # Parse input shape (handle dynamic shapes)
        try:
            # Try to get static shape
            raw_shape = model.input(0).shape
            parsed_shape = []
            for dim in raw_shape:
                if isinstance(dim, int):
                    parsed_shape.append(dim)
                else:
                    parsed_shape.append(None)  # dynamic dimension
            self._input_shape = tuple(parsed_shape)
        except RuntimeError:
            # If shape is fully dynamic, get partial shape
            partial_shape = model.input(0).get_partial_shape()
            parsed_shape = []
            for i in range(partial_shape.rank.get_length()):
                dim = partial_shape.get_dimension(i)
                if dim.is_static:
                    parsed_shape.append(dim.get_length())
                else:
                    parsed_shape.append(None)  # dynamic dimension
            self._input_shape = tuple(parsed_shape)
        
        # Check cache status before compilation
        cache_status = self._check_cache_status(onnx_path)
        print(f"[OpenVINO] {cache_status}")
        
        # Compile model with LATENCY hint
        print(f"[OpenVINO] Performance Hint: LATENCY")
        config = {"PERFORMANCE_HINT": "LATENCY"}
        
        try:
            self._compiled_model = self._core.compile_model(
                model=model, 
                device_name=device,
                config=config
            )
        except Exception as e:
            if device != "CPU":
                print(f"[OpenVINO] ⚠ {device} device unavailable, falling back to CPU")
                print(f"[OpenVINO] Error: {e}")
                self._compiled_model = self._core.compile_model(
                    model=model, 
                    device_name="CPU",
                    config=config
                )
                self._device = "CPU"
            else:
                raise RuntimeError(
                    f"Failed to compile model: {onnx_path}\n"
                    f"Device: {device}\n"
                    f"Performance Hint: LATENCY\n\n"
                    f"OpenVINO Error: {e}\n\n"
                    f"Possible causes:\n"
                    f"  - Incompatible model architecture\n"
                    f"  - Insufficient memory\n"
                    f"  - Corrupted model file\n\n"
                    f"Try:\n"
                    f"  - Verify ONNX file integrity\n"
                    f"  - Check system resources\n"
                    f"  - Update OpenVINO to latest version"
                ) from e
        
        print(f"[OpenVINO] Model compiled with LATENCY hint")
        
        # Create inference request
        self._infer_request = self._compiled_model.create_infer_request()
        
        # Save cache metadata after successful compilation
        if self._cache_enabled:
            try:
                self._save_cache_metadata(onnx_path)
            except Exception as e:
                print(f"[OpenVINO] ⚠ Failed to save cache metadata: {e}")
        
        self._backend = "openvino"
        self._loaded = True
        
        # Log which model was loaded
        model_name = self._get_model_name_from_path(onnx_path)
        print(f"[OpenVINO] Loaded {model_name}")

    def _check_cache_status(self, onnx_path: str) -> str:
        """
        Check if compiled model cache exists and is valid.
        
        Args:
            onnx_path: Path to source ONNX file
        
        Returns:
            Status string: "Cache hit" or "Cache miss"
        
        Cache validation:
            - Check if cache directory exists for this model
            - Compare ONNX file timestamp with cache metadata
            - Verify OpenVINO version matches
        """
        if not self._cache_enabled:
            return "Cache disabled"
        
        cache_key = self._generate_cache_key(onnx_path)
        cache_path = os.path.join(self._cache_dir, cache_key)
        meta_path = os.path.join(cache_path, "cache.meta")
        
        if not os.path.exists(cache_path):
            return "Cache miss"
        
        if not os.path.exists(meta_path):
            return "Cache miss"
        
        # Validate cache freshness
        try:
            import json
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            onnx_mtime = os.path.getmtime(onnx_path)
            cache_mtime = metadata.get('source_mtime', 0)
            
            if onnx_mtime > cache_mtime:
                return "Cache miss (source modified)"
            
            return "Cache hit"
        except Exception:
            return "Cache miss (validation failed)"

    def _generate_cache_key(self, model_path: str) -> str:
        """Generate unique cache key for model."""
        import hashlib
        try:
            from openvino import get_version
            ov_version = get_version()
        except:
            ov_version = "unknown"
        
        key_data = f"{os.path.abspath(model_path)}_{self._device}_LATENCY_{ov_version}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def _save_cache_metadata(self, source_path: str) -> None:
        """Save cache metadata for validation."""
        import json
        try:
            from openvino import get_version
            ov_version = get_version()
        except:
            ov_version = "unknown"
        
        cache_key = self._generate_cache_key(source_path)
        cache_path = os.path.join(self._cache_dir, cache_key)
        os.makedirs(cache_path, exist_ok=True)
        
        metadata = {
            'source_path': os.path.abspath(source_path),
            'source_mtime': os.path.getmtime(source_path),
            'openvino_version': ov_version,
            'device': self._device,
            'performance_hint': 'LATENCY',
            'created_at': time.time(),
            'cache_key': cache_key
        }
        
        meta_path = os.path.join(cache_path, "cache.meta")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _get_model_name_from_path(self, path: str) -> str:
        """Extract model identifier from path for logging."""
        if "model1" in path:
            return "Model 1"
        elif "model2" in path or "wlasl" in path.lower():
            return "Model 2"
        elif "model3" in path:
            return "Model 3"
        elif "model4" in path:
            return "Model 4"
        return "Model"

    def _is_model_openvino_compatible(self, onnx_path: str) -> bool:
        """
        Check if model is compatible with OpenVINO.
        
        Smart backend detection: Only Model 1 and Model 2 are compatible.
        Model 3 has unsupported Loop operator.
        
        Args:
            onnx_path: Path to ONNX model file
            
        Returns:
            True if model is compatible with OpenVINO, False otherwise
        """
        # Check path for model identifier
        if "model3" in onnx_path.lower():
            return False  # Model 3 is incompatible (Loop operator issue)
        
        # Model 1 and Model 2 are compatible
        if "model1" in onnx_path.lower() or "model2" in onnx_path.lower() or "wlasl" in onnx_path.lower():
            return True
        
        # Default: assume compatible and let OpenVINO try
        return True

    # ── ONNX Session ───────────────────────────

    def _load_onnx_session(self, onnx_path: str, use_gpu: bool = False) -> None:
        """
        Load ONNX model using ONNX Runtime.
        
        This is the fallback method for models incompatible with OpenVINO.
        Model 3 uses this path due to Loop operator incompatibility.
        
        Args:
            onnx_path: Path to ONNX model file
            use_gpu: If True, use GPU execution provider
        """
        import onnxruntime as ort
        
        print(f"[ONNX Runtime] Loading model: {onnx_path}")
        
        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set execution providers
        providers = []
        if use_gpu:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        print(f"[ONNX Runtime] Execution providers: {providers}")
        
        # Create inference session
        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output names and shapes
        input_meta = self._session.get_inputs()[0]
        output_meta = self._session.get_outputs()[0]
        
        self._input_name = input_meta.name
        self._output_name = output_meta.name
        self._input_shape = tuple(input_meta.shape)
        
        self._backend = BACKEND_ONNX
        self._loaded = True
        
        print(f"[ONNX Runtime] Model loaded successfully")
        print(f"[ONNX Runtime]   Input: {self._input_name} {self._input_shape}")
        print(f"[ONNX Runtime]   Output: {self._output_name} {output_meta.shape}")

    # ── SignBart Backend (dual-input, OpenVINO Runtime) ──────

    def _load_signbart(self, model_path: str) -> None:
        """
        Load a SignBart model (dual-input: keypoints + attention_mask) on the
        OpenVINO Runtime. Prefers a pre-converted IR (.xml) next to the given
        path; falls back to reading the ONNX directly through OpenVINO.

        SignBart expects:
          - keypoints      float32 (1, T, 75, 2)
          - attention_mask float32 (1, T)
        and returns logits (1, num_labels). The mask is built automatically
        (all ones) inside predict_raw from the keypoints' time dimension.
        """
        if self._core is None:
            self._initialize_openvino_core("CPU")

        # Prefer the IR if present (model_file may already be the .xml).
        ir_path = os.path.splitext(model_path)[0] + ".xml"
        load_target = ir_path if os.path.exists(ir_path) else model_path
        print(f"[SignBart] Loading model (OpenVINO): {load_target}")

        model = self._core.read_model(model=load_target)

        # Identify the two inputs by name: keypoints vs attention_mask.
        self._input_name = None
        self._mask_name = None
        for port in model.inputs:
            name = port.get_any_name()
            if "mask" in name.lower():
                self._mask_name = name
            else:
                self._input_name = name
        if self._input_name is None:
            self._input_name = model.inputs[0].get_any_name()
        if self._mask_name is None:
            self._mask_name = model.inputs[1].get_any_name()
        self._output_name = model.output(0).get_any_name()

        config = {"PERFORMANCE_HINT": "LATENCY"}
        self._compiled_model = self._core.compile_model(
            model=model, device_name=self._device, config=config
        )
        self._infer_request = self._compiled_model.create_infer_request()

        self._backend = BACKEND_SIGNBART
        self._loaded = True
        print(f"[SignBart] ✓ Loaded via OpenVINO — inputs: {self._input_name}, "
              f"{self._mask_name}; output: {self._output_name}")

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

        if self._backend == "openvino" and self._compiled_model:
            # Get output shape from OpenVINO compiled model
            outputs = self._compiled_model.outputs
            if outputs and len(outputs) > 0:
                try:
                    # Try to get static shape
                    shape = outputs[0].shape
                    if len(shape) >= 1 and isinstance(shape[-1], int):
                        num_output = shape[-1]
                except RuntimeError:
                    # Handle dynamic shape
                    partial_shape = outputs[0].get_partial_shape()
                    if partial_shape.rank.get_length() >= 1:
                        last_dim = partial_shape.get_dimension(partial_shape.rank.get_length() - 1)
                        if last_dim.is_static:
                            num_output = last_dim.get_length()

        elif self._backend == BACKEND_ONNX and self._session:
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
        Run raw inference using OpenVINO or ONNX Runtime.
        Returns the model's output tensor.
        Works with OpenVINO, ONNX, and Keras backends.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded — call load() first")

        if features.dtype != np.float32:
            features = features.astype(np.float32)

        if self._backend == BACKEND_SIGNBART:
            # features: (1, T, 75, 2). Build an all-ones attention mask (1, T).
            T = features.shape[1]
            mask = np.ones((features.shape[0], T), dtype=np.float32)
            result = self._infer_request.infer(
                inputs={self._input_name: features, self._mask_name: mask}
            )
            return np.array(result[self._output_name])

        if self._backend == "openvino":
            # Validate input shape
            expected_shape = self._input_shape
            if features.shape != expected_shape:
                # Handle dynamic dimensions (None) — only compare static dims
                same_rank = len(features.shape) == len(expected_shape)
                dims_ok = same_rank and all(
                    exp is None or exp == act
                    for exp, act in zip(expected_shape, features.shape)
                )
                if not dims_ok:
                    model_name = self._get_model_name_from_path(self._model_path)
                    raise ValueError(
                        f"Input shape mismatch for {model_name}.\n"
                        f"  Expected: {expected_shape} (None = dynamic dimension)\n"
                        f"  Received: {features.shape}\n\n"
                        f"Check the preprocessing pipeline produces the correct "
                        f"tensor shape before calling predict_raw()."
                    )
            
            # Run inference
            result = self._infer_request.infer(
                inputs={self._input_name: features}
            )
            
            # Extract output tensor
            output = result[self._output_name]
            
            # Convert to numpy array
            return np.array(output)

        elif self._backend == BACKEND_ONNX:
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
        if self._compiled_model:
            del self._compiled_model
            self._compiled_model = None
        if self._infer_request:
            del self._infer_request
            self._infer_request = None
        if self._core:
            del self._core
            self._core = None
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
