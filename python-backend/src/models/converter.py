"""
Model Format Converter
========================
Converts sign language models between formats:
  .h5 / .keras  →  .onnx   (via TensorFlow + tf2onnx)
  .tflite       →  .onnx   (via tf2onnx)

Converted models are cached alongside the original file
so conversion only happens once.

Also supports DIRECT Keras loading as fallback when ONNX
conversion fails (e.g., due to numpy version incompatibility).

Usage:
    from src.models.converter import ModelConverter
    
    onnx_path = ModelConverter.ensure_onnx("model.h5")
    onnx_path = ModelConverter.ensure_onnx("model.keras")
    onnx_path = ModelConverter.ensure_onnx("model.onnx")  # no-op
"""

import os
import time
from typing import Optional


class ModelConverter:
    """Handles conversion of various model formats to ONNX."""

    # Supported source formats
    SUPPORTED_EXTENSIONS = {".onnx", ".h5", ".keras", ".tflite"}

    @classmethod
    def ensure_onnx(cls, model_path: str) -> str:
        """
        Given a model file path, return the path to an ONNX version.
        
        - If already .onnx → returns as-is
        - If .h5/.keras/.tflite → checks for cached .onnx, converts if needed
        - Raises FileNotFoundError if source model doesn't exist
        - Raises RuntimeError if conversion fails
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        ext = os.path.splitext(model_path)[1].lower()

        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported model format: '{ext}'. "
                f"Supported: {', '.join(sorted(cls.SUPPORTED_EXTENSIONS))}"
            )

        # Already ONNX — nothing to do
        if ext == ".onnx":
            print(f"[Converter] Model is already ONNX: {model_path}")
            return model_path

        # Check for cached ONNX version
        onnx_path = os.path.splitext(model_path)[0] + ".onnx"
        if os.path.exists(onnx_path):
            # Verify cached version is newer than source
            if os.path.getmtime(onnx_path) >= os.path.getmtime(model_path):
                print(f"[Converter] Using cached ONNX: {onnx_path}")
                return onnx_path
            else:
                print(f"[Converter] Source model is newer — reconverting")

        # Convert based on source format
        if ext in (".h5", ".keras"):
            return cls._convert_keras_to_onnx(model_path, onnx_path)
        elif ext == ".tflite":
            return cls._convert_tflite_to_onnx(model_path, onnx_path)
        else:
            raise ValueError(f"No converter implemented for '{ext}'")

    # ── Keras (.h5 / .keras) → ONNX ────────────

    @classmethod
    def _convert_keras_to_onnx(cls, source_path: str, onnx_path: str) -> str:
        """Convert a Keras .h5 or .keras model to ONNX format."""
        print(f"[Converter] Converting Keras → ONNX: {source_path}")
        start = time.time()

        try:
            # Monkey-patch numpy for tf2onnx compatibility with NumPy 2.x
            cls._patch_numpy_compat()

            import tensorflow as tf
            import tf2onnx
        except ImportError as e:
            raise RuntimeError(
                f"Cannot convert Keras model — missing dependency: {e}\n"
                f"Install with: pip install tensorflow tf2onnx\n"
                f"Or manually convert your model to .onnx format."
            ) from e

        try:
            # Load the Keras model
            print(f"[Converter]   Loading Keras model...")
            model = tf.keras.models.load_model(source_path, compile=False)

            # Log model info
            input_shape = model.input_shape
            output_shape = model.output_shape
            print(f"[Converter]   Input shape:  {input_shape}")
            print(f"[Converter]   Output shape: {output_shape}")
            print(f"[Converter]   Parameters:   {model.count_params():,}")

            # Build input spec from model's input shape
            if isinstance(input_shape, list):
                input_spec = [
                    tf.TensorSpec(
                        shape=[None if d is None else d for d in shape],
                        dtype=tf.float32,
                        name=f"input_{i}"
                    )
                    for i, shape in enumerate(input_shape)
                ]
            else:
                input_spec = [
                    tf.TensorSpec(
                        shape=[None if d is None else d for d in input_shape],
                        dtype=tf.float32,
                        name="input"
                    )
                ]

            # Convert to ONNX
            print(f"[Converter]   Converting with tf2onnx...")
            model_proto, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=input_spec,
                opset=13,
                output_path=onnx_path,
            )

            elapsed = time.time() - start
            file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(
                f"[Converter] ✓ Conversion complete in {elapsed:.1f}s "
                f"— {file_size_mb:.1f} MB → {onnx_path}"
            )
            return onnx_path

        except Exception as e:
            # Clean up partial ONNX file on failure
            if os.path.exists(onnx_path):
                try:
                    os.remove(onnx_path)
                except OSError:
                    pass
            raise RuntimeError(f"Keras → ONNX conversion failed: {e}") from e

    # ── TFLite → ONNX ──────────────────────────

    @classmethod
    def _convert_tflite_to_onnx(cls, source_path: str, onnx_path: str) -> str:
        """Convert a TFLite model to ONNX format."""
        print(f"[Converter] Converting TFLite → ONNX: {source_path}")
        start = time.time()

        try:
            cls._patch_numpy_compat()
            import tf2onnx
        except ImportError as e:
            raise RuntimeError(
                f"Cannot convert TFLite model — missing dependency: {e}\n"
                f"Install with: pip install tf2onnx\n"
                f"Or manually convert your model to .onnx format."
            ) from e

        try:
            print(f"[Converter]   Converting with tf2onnx...")
            model_proto, _ = tf2onnx.convert.from_tflite(
                source_path,
                opset=13,
                output_path=onnx_path,
            )

            elapsed = time.time() - start
            file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(
                f"[Converter] ✓ Conversion complete in {elapsed:.1f}s "
                f"— {file_size_mb:.1f} MB → {onnx_path}"
            )
            return onnx_path

        except Exception as e:
            if os.path.exists(onnx_path):
                try:
                    os.remove(onnx_path)
                except OSError:
                    pass
            raise RuntimeError(f"TFLite → ONNX conversion failed: {e}") from e

    # ── Keras Direct Loading (Fallback) ─────────

    @classmethod
    def load_keras_model(cls, model_path: str):
        """
        Load a Keras model directly (fallback when ONNX conversion fails).
        Returns the loaded Keras model.
        """
        print(f"[Converter] Loading Keras model directly: {model_path}")
        start = time.time()

        try:
            import keras
            model = keras.models.load_model(model_path, compile=False)
            elapsed = time.time() - start
            print(
                f"[Converter] ✓ Keras model loaded in {elapsed:.1f}s "
                f"— input: {model.input_shape}, output: {model.output_shape}"
            )
            return model
        except ImportError:
            pass

        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path, compile=False)
            elapsed = time.time() - start
            print(
                f"[Converter] ✓ Keras model loaded (via tf) in {elapsed:.1f}s "
                f"— input: {model.input_shape}, output: {model.output_shape}"
            )
            return model
        except ImportError as e:
            raise RuntimeError(
                f"Cannot load Keras model — neither keras nor tensorflow installed: {e}\n"
                f"Install with: pip install keras  OR  pip install tensorflow"
            ) from e

    # ── NumPy Compatibility ─────────────────────

    @staticmethod
    def _patch_numpy_compat():
        """
        Monkey-patch numpy for compatibility with tf2onnx on NumPy 2.x.
        
        tf2onnx uses np.object, np.bool, np.str etc. which were removed
        in NumPy 1.24+. This patch restores them as aliases to builtins.
        """
        import numpy as np

        compat_attrs = {
            "object": object,
            "bool": bool,
            "str": str,
            "int": int,
            "float": float,
            "complex": complex,
        }

        for attr, builtin in compat_attrs.items():
            if not hasattr(np, attr):
                setattr(np, attr, builtin)

    # ── Utilities ───────────────────────────────

    @classmethod
    def detect_format(cls, model_path: str) -> Optional[str]:
        """Detect the model format from the file extension."""
        ext = os.path.splitext(model_path)[1].lower()
        if ext in cls.SUPPORTED_EXTENSIONS:
            return ext
        return None

    @classmethod
    def find_model_file(cls, directory: str) -> Optional[str]:
        """
        Scan a directory for the first supported model file.
        Priority: .onnx > .h5 > .keras > .tflite
        """
        if not os.path.isdir(directory):
            return None

        priority = [".onnx", ".h5", ".keras", ".tflite"]
        files_by_ext: dict[str, list[str]] = {ext: [] for ext in priority}

        for filename in os.listdir(directory):
            ext = os.path.splitext(filename)[1].lower()
            if ext in files_by_ext:
                files_by_ext[ext].append(os.path.join(directory, filename))

        for ext in priority:
            if files_by_ext[ext]:
                chosen = files_by_ext[ext][0]
                print(f"[Converter] Auto-discovered model: {chosen}")
                return chosen

        return None

    @classmethod
    def get_model_info(cls, model_path: str) -> dict:
        """Get basic info about a model file without loading it."""
        ext = os.path.splitext(model_path)[1].lower()
        size_bytes = os.path.getsize(model_path) if os.path.exists(model_path) else 0

        return {
            "path": model_path,
            "filename": os.path.basename(model_path),
            "format": ext.lstrip(".").upper(),
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "exists": os.path.exists(model_path),
            "has_cached_onnx": (
                ext != ".onnx"
                and os.path.exists(os.path.splitext(model_path)[0] + ".onnx")
            ),
        }
