#!/usr/bin/env python3
"""
SignSpeak — Model Conversion Script
======================================
Converts .h5 / .keras / .tflite models to ONNX format
for faster inference with ONNX Runtime.

Usage:
    python convert_model.py models/sign/model5.keras
    python convert_model.py models/sign/model5.h5
    python convert_model.py models/sign/model.h5 --info
    python convert_model.py models/sign/model.onnx --info

Prerequisites:
    pip install tensorflow tf2onnx onnxruntime

If you hit numpy errors during conversion:
    pip install numpy<2.0
    python convert_model.py models/sign/model.h5
    pip install numpy>=2.0  # restore after conversion
"""

import os
import sys
import argparse


def patch_numpy():
    """Fix numpy 2.x compatibility for tf2onnx."""
    import numpy as np
    for attr, builtin in {
        "object": object, "bool": bool, "str": str,
        "int": int, "float": float, "complex": complex,
    }.items():
        if not hasattr(np, attr):
            setattr(np, attr, builtin)


def show_model_info(model_path: str):
    """Print model architecture and shapes."""
    ext = os.path.splitext(model_path)[1].lower()

    if ext in (".h5", ".keras"):
        try:
            import keras
            model = keras.models.load_model(model_path, compile=False)
        except ImportError:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path, compile=False)

        print(f"\n{'=' * 60}")
        print(f"  Model: {os.path.basename(model_path)}")
        print(f"  Format: {ext.upper()}")
        print(f"  Input shape:  {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Parameters:   {model.count_params():,}")
        print(f"{'=' * 60}")
        print()
        model.summary()

    elif ext == ".onnx":
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path)
            inputs = session.get_inputs()
            outputs = session.get_outputs()

            print(f"\n{'=' * 60}")
            print(f"  Model: {os.path.basename(model_path)}")
            print(f"  Format: ONNX")
            print(f"  Size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
            print(f"  Provider: {session.get_providers()[0]}")
            print(f"  Inputs:")
            for inp in inputs:
                print(f"    - {inp.name}: {inp.shape} ({inp.type})")
            print(f"  Outputs:")
            for out in outputs:
                print(f"    - {out.name}: {out.shape} ({out.type})")
            print(f"{'=' * 60}")
        except ImportError:
            print("Install onnxruntime to inspect ONNX models: pip install onnxruntime")

    elif ext == ".tflite":
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            inputs = interpreter.get_input_details()
            outputs = interpreter.get_output_details()

            print(f"\n{'=' * 60}")
            print(f"  Model: {os.path.basename(model_path)}")
            print(f"  Format: TFLite")
            print(f"  Inputs:")
            for inp in inputs:
                print(f"    - {inp['name']}: {inp['shape']} ({inp['dtype']})")
            print(f"  Outputs:")
            for out in outputs:
                print(f"    - {out['name']}: {out['shape']} ({out['dtype']})")
            print(f"{'=' * 60}")
        except ImportError:
            print("Install tensorflow to inspect TFLite models: pip install tensorflow")
    else:
        print(f"Unsupported format: {ext}")


def convert_to_onnx(model_path: str):
    """Convert model to ONNX format."""
    ext = os.path.splitext(model_path)[1].lower()

    if ext == ".onnx":
        print(f"Already ONNX: {model_path}")
        return

    if ext not in (".h5", ".keras", ".tflite"):
        print(f"Unsupported format for conversion: {ext}")
        sys.exit(1)

    # Patch numpy before importing tf2onnx
    patch_numpy()

    # Use our converter
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.models.converter import ModelConverter

    try:
        onnx_path = ModelConverter.ensure_onnx(model_path)
        print(f"\n✓ Converted successfully: {onnx_path}")
        print(f"  Size: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")

        # Verify the converted model loads
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path)
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            print(f"  Input:  {inputs[0].name} {inputs[0].shape}")
            print(f"  Output: {outputs[0].name} {outputs[0].shape}")
            print(f"  ✓ ONNX model verified!")
        except Exception as e:
            print(f"  ⚠ Could not verify ONNX model: {e}")

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Install dependencies: pip install tensorflow tf2onnx")
        print(f"  2. If numpy error: pip install 'numpy<2.0', convert, then pip install 'numpy>=2.0'")
        print(f"  3. Or use Keras fallback (no conversion needed — just place .h5/.keras in models/sign/)")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="SignSpeak Model Converter — .h5/.keras/.tflite → ONNX"
    )
    parser.add_argument(
        "model_path",
        help="Path to the model file (.h5, .keras, .tflite, .onnx)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print model info instead of converting"
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"File not found: {args.model_path}")
        sys.exit(1)

    if args.info:
        show_model_info(args.model_path)
    else:
        convert_to_onnx(args.model_path)


if __name__ == "__main__":
    main()
