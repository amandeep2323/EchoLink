#!/usr/bin/env python3
"""
Model3 ONNX Conversion Script
==============================
Converts the LSTM-based ASL recognition model from jamesjbustos/sign-language-recognition
repository to ONNX format for integration into the desktop application.

Usage:
    python convert_model3.py <input_model_path> [--output-dir <dir>] [--verify]

Arguments:
    input_model_path: Path to original model file (.h5, .keras, or .tflite)
    --output-dir: Directory to save ONNX model (default: models/sign/model3/)
    --verify: Run verification tests after conversion (default: True)
    --no-verify: Skip verification tests

Requirements:
    - tensorflow >= 2.10
    - tf2onnx >= 1.13
    - onnxruntime >= 1.13
    - numpy

Example:
    python convert_model3.py models/sign/model3/staging/model.tflite
    python convert_model3.py model3_lstm.h5 --output-dir models/sign/model3/

Output:
    - models/sign/model3/model.onnx: Converted ONNX model
    - Verification report printed to console
"""

import os
import sys
import argparse
import time
from typing import Tuple, Optional


def patch_numpy_compat():
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


def load_tflite_model(model_path: str) -> Tuple[object, dict, dict]:
    """
    Load a TFLite model and extract metadata.
    
    Args:
        model_path: Path to .tflite model file
        
    Returns:
        Tuple of (interpreter, input_details, output_details)
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise RuntimeError(
            f"Cannot load TFLite model — TensorFlow not installed: {e}\n"
            f"Install with: pip install tensorflow"
        ) from e
    
    print(f"[Loader] Loading TFLite model: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"[Loader] Input:  {input_details['name']} {input_details['shape']} ({input_details['dtype']})")
    print(f"[Loader] Output: {output_details['name']} {output_details['shape']} ({output_details['dtype']})")
    
    return interpreter, input_details, output_details


def load_keras_model(model_path: str) -> Tuple[object, tuple, tuple]:
    """
    Load a Keras (.h5 or .keras) model and extract metadata.
    
    Args:
        model_path: Path to .h5 or .keras model file
        
    Returns:
        Tuple of (model, input_shape, output_shape)
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise RuntimeError(
            f"Cannot load Keras model — TensorFlow not installed: {e}\n"
            f"Install with: pip install tensorflow"
        ) from e
    
    print(f"[Loader] Loading Keras model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    input_shape = model.input_shape
    output_shape = model.output_shape
    
    print(f"[Loader] Input shape:  {input_shape}")
    print(f"[Loader] Output shape: {output_shape}")
    print(f"[Loader] Parameters:   {model.count_params():,}")
    
    return model, input_shape, output_shape


def convert_tflite_to_onnx(tflite_path: str, onnx_path: str) -> str:
    """
    Convert a TFLite model to ONNX format.
    
    Args:
        tflite_path: Path to source .tflite file
        onnx_path: Path to save converted .onnx file
        
    Returns:
        Path to converted ONNX model
        
    Raises:
        RuntimeError: If conversion fails
    """
    print(f"\n{'=' * 60}")
    print(f"  Converting TFLite → ONNX")
    print(f"{'=' * 60}")
    print(f"  Source: {tflite_path}")
    print(f"  Target: {onnx_path}")
    print()
    
    start = time.time()
    
    try:
        patch_numpy_compat()
        import tf2onnx
    except ImportError as e:
        raise RuntimeError(
            f"Cannot convert TFLite model — missing dependency: {e}\n"
            f"Install with: pip install tf2onnx\n"
            f"Or manually convert your model to .onnx format."
        ) from e
    
    try:
        print(f"[Converter] Converting with tf2onnx (opset 13)...")
        model_proto, _ = tf2onnx.convert.from_tflite(
            tflite_path,
            opset=13,
            output_path=onnx_path,
        )
        
        elapsed = time.time() - start
        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        
        print()
        print(f"{'=' * 60}")
        print(f"  ✓ Conversion Complete")
        print(f"{'=' * 60}")
        print(f"  Time:   {elapsed:.1f}s")
        print(f"  Size:   {file_size_mb:.2f} MB")
        print(f"  Output: {onnx_path}")
        print(f"{'=' * 60}")
        print()
        
        return onnx_path
        
    except Exception as e:
        # Clean up partial ONNX file on failure
        if os.path.exists(onnx_path):
            try:
                os.remove(onnx_path)
            except OSError:
                pass
        raise RuntimeError(f"TFLite → ONNX conversion failed: {e}") from e


def convert_keras_to_onnx(keras_path: str, onnx_path: str) -> str:
    """
    Convert a Keras (.h5 or .keras) model to ONNX format.
    
    Args:
        keras_path: Path to source .h5 or .keras file
        onnx_path: Path to save converted .onnx file
        
    Returns:
        Path to converted ONNX model
        
    Raises:
        RuntimeError: If conversion fails
    """
    print(f"\n{'=' * 60}")
    print(f"  Converting Keras → ONNX")
    print(f"{'=' * 60}")
    print(f"  Source: {keras_path}")
    print(f"  Target: {onnx_path}")
    print()
    
    start = time.time()
    
    try:
        patch_numpy_compat()
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
        print(f"[Converter] Loading Keras model...")
        model = tf.keras.models.load_model(keras_path, compile=False)
        
        # Log model info
        input_shape = model.input_shape
        output_shape = model.output_shape
        print(f"[Converter] Input shape:  {input_shape}")
        print(f"[Converter] Output shape: {output_shape}")
        print(f"[Converter] Parameters:   {model.count_params():,}")
        
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
        print(f"[Converter] Converting with tf2onnx (opset 13)...")
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_spec,
            opset=13,
            output_path=onnx_path,
        )
        
        elapsed = time.time() - start
        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        
        print()
        print(f"{'=' * 60}")
        print(f"  ✓ Conversion Complete")
        print(f"{'=' * 60}")
        print(f"  Time:   {elapsed:.1f}s")
        print(f"  Size:   {file_size_mb:.2f} MB")
        print(f"  Output: {onnx_path}")
        print(f"{'=' * 60}")
        print()
        
        return onnx_path
        
    except Exception as e:
        # Clean up partial ONNX file on failure
        if os.path.exists(onnx_path):
            try:
                os.remove(onnx_path)
            except OSError:
                pass
        raise RuntimeError(f"Keras → ONNX conversion failed: {e}") from e


def verify_onnx_model(onnx_path: str, original_path: str) -> bool:
    """
    Verify the converted ONNX model loads and produces valid outputs.
    
    Args:
        onnx_path: Path to converted ONNX model
        original_path: Path to original model (for comparison)
        
    Returns:
        True if verification passes, False otherwise
    """
    print(f"\n{'=' * 60}")
    print(f"  Verifying ONNX Model")
    print(f"{'=' * 60}")
    
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
    except ImportError as e:
        print(f"⚠ Cannot verify — missing dependency: {e}")
        print(f"  Install with: pip install onnx onnxruntime")
        return False
    
    try:
        # Load and validate ONNX model
        print(f"[Verify] Loading ONNX model...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"[Verify] ✓ ONNX model validation passed")
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Get input/output metadata
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        
        print(f"[Verify] Input:  {input_meta.name} {input_meta.shape} ({input_meta.type})")
        print(f"[Verify] Output: {output_meta.name} {output_meta.shape} ({output_meta.type})")
        
        # Create dummy input matching expected shape
        input_shape = input_meta.shape
        # Replace dynamic dimensions (None, -1) with 1
        test_shape = tuple(1 if (d is None or d == -1 or d == 'batch_size') else d for d in input_shape)
        dummy_input = np.random.rand(*test_shape).astype(np.float32)
        
        print(f"[Verify] Test input shape: {dummy_input.shape}")
        
        # Run inference
        onnx_output = session.run([output_meta.name], {input_meta.name: dummy_input})[0]
        print(f"[Verify] Test output shape: {onnx_output.shape}")
        
        # Check output validity
        if np.any(np.isnan(onnx_output)):
            print(f"[Verify] ✗ Output contains NaN values")
            return False
        
        if np.any(np.isinf(onnx_output)):
            print(f"[Verify] ✗ Output contains Inf values")
            return False
        
        print(f"[Verify] Output range: [{np.min(onnx_output):.4f}, {np.max(onnx_output):.4f}]")
        print(f"[Verify] Output sample (first 5): {onnx_output[0][:5]}")
        
        # Compare with original model if possible
        ext = os.path.splitext(original_path)[1].lower()
        
        if ext == '.tflite':
            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=original_path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()[0]
                output_details = interpreter.get_output_details()[0]
                
                # Run inference on original model
                interpreter.set_tensor(input_details['index'], dummy_input)
                interpreter.invoke()
                original_output = interpreter.get_tensor(output_details['index'])
                
                # Compare outputs
                max_diff = np.max(np.abs(original_output - onnx_output))
                print(f"[Verify] Max difference (TFLite vs ONNX): {max_diff:.8f}")
                
                if max_diff < 0.001:
                    print(f"[Verify] ✓ ONNX predictions match TFLite — conversion accurate!")
                else:
                    print(f"[Verify] ⚠ Predictions differ by {max_diff:.6f} — review carefully")
                    
            except Exception as e:
                print(f"[Verify] ⚠ Could not compare with original: {e}")
        
        elif ext in ('.h5', '.keras'):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(original_path, compile=False)
                
                # Run inference on original model
                original_output = model.predict(dummy_input, verbose=0)
                
                # Compare outputs
                max_diff = np.max(np.abs(original_output - onnx_output))
                print(f"[Verify] Max difference (Keras vs ONNX): {max_diff:.8f}")
                
                if max_diff < 0.001:
                    print(f"[Verify] ✓ ONNX predictions match Keras — conversion accurate!")
                else:
                    print(f"[Verify] ⚠ Predictions differ by {max_diff:.6f} — review carefully")
                    
            except Exception as e:
                print(f"[Verify] ⚠ Could not compare with original: {e}")
        
        print()
        print(f"{'=' * 60}")
        print(f"  ✓ Verification Complete")
        print(f"{'=' * 60}")
        print()
        
        return True
        
    except Exception as e:
        print(f"[Verify] ✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Model3 ONNX Conversion Script — Convert LSTM ASL model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_model3.py models/sign/model3/staging/model.tflite
  python convert_model3.py model3_lstm.h5 --output-dir models/sign/model3/
  python convert_model3.py model.keras --no-verify

Requirements:
  pip install tensorflow tf2onnx onnxruntime numpy
        """
    )
    
    parser.add_argument(
        "input_path",
        help="Path to original model file (.h5, .keras, or .tflite)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="models/sign/model3/",
        help="Directory to save ONNX model (default: models/sign/model3/)"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run verification tests after conversion (default: True)"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_false",
        dest="verify",
        help="Skip verification tests"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_path):
        print(f"✗ Error: Input file not found: {args.input_path}")
        sys.exit(1)
    
    # Detect input format
    ext = os.path.splitext(args.input_path)[1].lower()
    if ext not in ('.h5', '.keras', '.tflite'):
        print(f"✗ Error: Unsupported format '{ext}'")
        print(f"  Supported formats: .h5, .keras, .tflite")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output path
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    
    # Check if output already exists
    if os.path.exists(onnx_path):
        print(f"⚠ Warning: Output file already exists: {onnx_path}")
        response = input("  Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("  Conversion cancelled.")
            sys.exit(0)
    
    print(f"\n{'=' * 60}")
    print(f"  Model3 ONNX Conversion")
    print(f"{'=' * 60}")
    print(f"  Input:  {args.input_path}")
    print(f"  Output: {onnx_path}")
    print(f"  Format: {ext.upper()} → ONNX")
    print(f"{'=' * 60}\n")
    
    try:
        # Convert based on input format
        if ext == '.tflite':
            # Load TFLite model to show info
            load_tflite_model(args.input_path)
            # Convert to ONNX
            convert_tflite_to_onnx(args.input_path, onnx_path)
        elif ext in ('.h5', '.keras'):
            # Load Keras model to show info
            load_keras_model(args.input_path)
            # Convert to ONNX
            convert_keras_to_onnx(args.input_path, onnx_path)
        
        # Verify conversion if requested
        if args.verify:
            success = verify_onnx_model(onnx_path, args.input_path)
            if not success:
                print("⚠ Warning: Verification encountered issues")
                print("  Review the output carefully before using the model")
        
        print(f"✓ Conversion complete!")
        print(f"  ONNX model saved to: {onnx_path}")
        print()
        
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Install dependencies: pip install tensorflow tf2onnx onnxruntime")
        print(f"  2. Check TensorFlow version: pip install 'tensorflow>=2.10'")
        print(f"  3. If numpy error: pip install 'numpy<2.0', convert, then pip install 'numpy>=2.0'")
        sys.exit(1)


if __name__ == "__main__":
    main()
