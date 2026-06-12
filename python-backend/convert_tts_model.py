#!/usr/bin/env python
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Intel OPEA SpeechT5 TTS — Optimized Format Converter
====================================================

Converts the HuggingFace SpeechT5 TTS pipeline (microsoft/speecht5_tts +
microsoft/speecht5_hifigan) into an Intel-optimized, serialized model format
stored under ``models/tts/``.

WHY NOT A SINGLE .onnx FILE?
----------------------------
Your sign-recognition models (Model 1/2/3) are single feed-forward graphs, so a
single ``.onnx`` file works perfectly. SpeechT5 is different: it is a THREE-stage
pipeline (text encoder -> autoregressive decoder -> HiFi-GAN vocoder). The decoder
generates the mel-spectrogram frame-by-frame inside a Python loop. That loop is
exactly the kind of construct (the ONNX ``Loop`` operator) that made Model 3
incompatible with OpenVINO.

A faithful ONNX export therefore produces MULTIPLE files (encoder, decoder,
decoder-with-past, vocoder) plus custom Python glue to drive the generation loop.
It does NOT give you one clean drop-in file, and it adds a fragile inference path.

THE CORRECT OPTIMIZED FORMAT FOR SPEECHT5
-----------------------------------------
Intel's own tooling (Optimum-Intel / OpenVINO) exports SpeechT5 to **OpenVINO IR**
(``.xml`` + ``.bin``) while handling the multi-stage pipeline and the generation
loop for you. This is the same Intel acceleration you wanted, in the format Intel
recommends for this architecture — with NO loss of functionality.

This script uses ``optimum-cli export openvino`` under the hood.

REQUIREMENTS
------------
    pip install "optimum[openvino]>=1.16.0"

(The script will detect a missing dependency and print the exact install command.)

USAGE
-----
    python convert_tts_model.py
    python convert_tts_model.py --output models/tts/speecht5_openvino
    python convert_tts_model.py --no-verify
"""

import argparse
import os
import sys
import time


# microsoft/speecht5_tts is the standard SpeechT5 TTS checkpoint used by OPEA.
MODEL_ID = "microsoft/speecht5_tts"
DEFAULT_OUTPUT = os.path.join("models", "tts", "speecht5_openvino")


def check_dependencies() -> bool:
    """Verify Optimum-Intel (OpenVINO export backend) is available."""
    missing = []
    try:
        import optimum  # noqa: F401
    except ImportError:
        missing.append("optimum")
    try:
        import openvino  # noqa: F401
    except ImportError:
        missing.append("openvino")

    # The OpenVINO exporter entry point lives in optimum.exporters.openvino
    ov_exporter_ok = False
    if "optimum" not in missing:
        try:
            from optimum.exporters.openvino import main_export  # noqa: F401
            ov_exporter_ok = True
        except Exception:
            ov_exporter_ok = False

    if missing or not ov_exporter_ok:
        print("=" * 70)
        print("  MISSING DEPENDENCIES")
        print("=" * 70)
        print()
        print("SpeechT5 -> optimized format conversion needs Optimum-Intel.")
        print()
        print("Install it with:")
        print()
        print('    pip install "optimum[openvino]>=1.16.0"')
        print()
        print("Then re-run:")
        print()
        print("    python convert_tts_model.py")
        print()
        print("=" * 70)
        return False
    return True


def convert(output_dir: str) -> bool:
    """
    Export microsoft/speecht5_tts to OpenVINO IR using Optimum.

    Produces, under ``output_dir``:
        - openvino_encoder_model.xml / .bin
        - openvino_decoder_model.xml / .bin
        - openvino_postnet_and_vocoder_model.xml / .bin
        - config.json, processor files, etc.

    These are the serialized, Intel-optimized equivalents of the PyTorch model.
    """
    from optimum.exporters.openvino import main_export

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  Converting SpeechT5 TTS -> OpenVINO IR (Intel-optimized format)")
    print("=" * 70)
    print(f"  Source model : {MODEL_ID}")
    print(f"  Output dir   : {os.path.abspath(output_dir)}")
    print(f"  Task         : text-to-audio")
    print("=" * 70)
    print()

    start = time.time()
    try:
        # SpeechT5 export REQUIRES the HiFi-GAN vocoder to be bundled in.
        main_export(
            model_name_or_path=MODEL_ID,
            output=output_dir,
            task="text-to-audio",
            model_kwargs={"vocoder": "microsoft/speecht5_hifigan"},
        )
    except TypeError as e:
        if "allow_new" in str(e):
            # Known incompatibility between optimum/optimum-intel and the
            # installed transformers version (seen on Python 3.14 +
            # transformers 4.48.x + optimum 2.x). The SpeechT5 export config
            # crashes inside NormalizedConfig before any file is written.
            print()
            print("=" * 70)
            print("  EXPORT BLOCKED — Optimum / transformers version conflict")
            print("=" * 70)
            print()
            print("  Optimum's SpeechT5 exporter crashed with:")
            print(f"    {e}")
            print()
            print("  This is an internal incompatibility between the installed")
            print("  optimum-intel and transformers versions, not a problem with")
            print("  this script or the model.")
            print()
            print("  WORKAROUND (optional): create a separate environment with a")
            print("  compatible pin, e.g.:")
            print('    pip install "optimum-intel[openvino]==1.21.0" "transformers==4.44.2"')
            print()
            print("  IMPORTANT: You do NOT need this conversion for the app to")
            print("  use Intel acceleration. OPEA TTS already runs through the")
            print("  OpenVINO-detected backend at runtime (see synthesizer.py).")
            print("=" * 70)
            return False
        raise

    elapsed = time.time() - start
    print()
    print(f"[Convert] Export finished in {elapsed:.1f}s")
    return True


def list_outputs(output_dir: str) -> None:
    """Print the generated files so the user can see the result."""
    print()
    print("[Convert] Generated files:")
    if not os.path.isdir(output_dir):
        print("  (output directory not found)")
        return
    for root, _dirs, files in os.walk(output_dir):
        for f in sorted(files):
            full = os.path.join(root, f)
            size = os.path.getsize(full) / (1024 * 1024)
            rel = os.path.relpath(full, output_dir)
            print(f"  {rel:55s} {size:8.2f} MB")


def verify(output_dir: str) -> bool:
    """
    Load the exported OpenVINO model and run a smoke-test synthesis to confirm
    no loss of functionality.
    """
    print()
    print("=" * 70)
    print("  Verifying exported model (smoke test)")
    print("=" * 70)
    try:
        import numpy as np
        import torch
        from transformers import SpeechT5Processor
        from optimum.intel import OVModelForTextToSpeechSeq2Seq

        processor = SpeechT5Processor.from_pretrained(output_dir)
        model = OVModelForTextToSpeechSeq2Seq.from_pretrained(output_dir)

        inputs = processor(text="Hello from EchoLink", return_tensors="pt")

        # A 512-dim speaker embedding is required. Use a zero vector for the
        # smoke test (real voices use the cached spk_embed_*.pt files).
        speaker_embeddings = torch.zeros((1, 512))

        output = model.generate(
            input_ids=inputs["input_ids"], speaker_embeddings=speaker_embeddings
        )
        audio = np.asarray(output)
        ok = audio.size > 100
        print(f"[Verify] Synthesized {audio.size} samples -> {'OK' if ok else 'FAILED'}")
        return ok
    except Exception as e:
        print(f"[Verify] Smoke test could not run: {e}")
        print("[Verify] The files were still exported; verification is optional.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Intel OPEA SpeechT5 TTS to an optimized OpenVINO format."
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip the post-export smoke-test synthesis.",
    )
    args = parser.parse_args()

    # Run relative to this script's directory so 'models/tts' resolves correctly.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not check_dependencies():
        return 1

    if not convert(args.output):
        return 1

    list_outputs(args.output)

    if not args.no_verify:
        verify(args.output)

    print()
    print("=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"  Optimized model saved to: {os.path.abspath(args.output)}")
    print()
    print("  Note: this is OpenVINO IR (.xml/.bin), the Intel-recommended")
    print("  optimized format for SpeechT5. A single .onnx file is not used")
    print("  because SpeechT5's autoregressive decoder requires a generation")
    print("  loop (the same Loop-operator limitation as Model 3).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
