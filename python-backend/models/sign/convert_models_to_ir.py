#!/usr/bin/env python
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Sign Recognition Models -> OpenVINO IR Converter
=================================================

Converts the sign-recognition ONNX models (Model 1, Model 2, Model 3) into the
Intel-optimized OpenVINO IR format (.xml + .bin), placing each IR file next to
its source ONNX inside the model's own folder.

BEHAVIOUR
---------
For each model directory (model1, model2, model3):
  1. Read model.json to find the source ONNX file.
  2. Attempt conversion with openvino.convert_model().
  3. On success: save <name>.xml / <name>.bin in the SAME folder.
  4. On failure (e.g. unsupported operators like the LSTM 'Loop'): leave the
     model in ONNX format and report it. The app falls back to ONNX Runtime
     automatically for these.

This mirrors the runtime's hybrid strategy:
  - Model 1 (PointNet)      -> convertible -> OpenVINO IR
  - Model 2 (WLASL Pose-TGCN)-> convertible -> OpenVINO IR
  - Model 3 (LSTM)          -> NOT convertible (Loop op) -> stays ONNX

USAGE
-----
    python convert_models_to_ir.py
    python convert_models_to_ir.py --check-only      # Task 1: feasibility only
    python convert_models_to_ir.py --force           # re-convert even if IR exists
"""

import argparse
import json
import os
import sys
import time


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Model folders to process, with expected output class counts for a sanity check.
MODELS = [
    {"id": "model1", "dir": os.path.join(THIS_DIR, "model1"), "classes": 24},
    {"id": "model2", "dir": os.path.join(THIS_DIR, "model2"), "classes": 2000},
    {"id": "model3", "dir": os.path.join(THIS_DIR, "model3"), "classes": 250},
]


def _load_onnx_file_name(model_dir: str) -> str:
    """Return the ONNX file name from model.json, defaulting to model.onnx."""
    config_path = os.path.join(model_dir, "model.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            mf = cfg.get("model_file", "model.onnx")
            if mf.lower().endswith(".onnx"):
                return mf
        except Exception:
            pass
    return "model.onnx"


def check_convertible(core, onnx_path: str) -> tuple:
    """
    Task 1: check whether an ONNX model can be converted to OpenVINO IR.

    Uses openvino.convert_model() which performs the full graph conversion.
    Returns (ok: bool, detail: str).
    """
    import openvino as ov

    try:
        ov_model = ov.convert_model(onnx_path)
        # Touch shapes to confirm the graph is well-formed.
        n_in = len(ov_model.inputs)
        n_out = len(ov_model.outputs)
        return True, f"convertible (inputs={n_in}, outputs={n_out})"
    except Exception as e:
        msg = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
        return False, f"NOT convertible: {msg}"


def convert_one(core, model_id: str, model_dir: str, force: bool) -> dict:
    """
    Convert a single model's ONNX to OpenVINO IR (if possible) and save it
    in the same folder. Returns a result dict.
    """
    import openvino as ov

    result = {"id": model_id, "status": "unknown", "ir": None, "detail": ""}

    onnx_name = _load_onnx_file_name(model_dir)
    onnx_path = os.path.join(model_dir, onnx_name)
    base = os.path.splitext(onnx_name)[0]
    xml_path = os.path.join(model_dir, base + ".xml")
    bin_path = os.path.join(model_dir, base + ".bin")

    if not os.path.exists(onnx_path):
        result["status"] = "missing"
        result["detail"] = f"ONNX not found: {onnx_path}"
        print(f"[{model_id}] ✗ {result['detail']}")
        return result

    print(f"\n[{model_id}] Source ONNX: {onnx_name}")

    # Skip if IR already present and not forcing.
    if os.path.exists(xml_path) and os.path.exists(bin_path) and not force:
        result["status"] = "exists"
        result["ir"] = base + ".xml"
        result["detail"] = "IR already present (use --force to rebuild)"
        print(f"[{model_id}] ✓ {result['detail']}: {base}.xml")
        return result

    # Attempt conversion.
    print(f"[{model_id}] Converting to OpenVINO IR...")
    try:
        ov_model = ov.convert_model(onnx_path)
    except Exception as e:
        msg = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
        result["status"] = "incompatible"
        result["detail"] = msg
        print(f"[{model_id}] ✗ Cannot convert to IR — keeping ONNX format")
        print(f"[{model_id}]   Reason: {msg}")
        return result

    # Save IR alongside the ONNX file.
    try:
        ov.save_model(ov_model, xml_path)
        size_mb = (os.path.getsize(bin_path) / (1024 * 1024)) if os.path.exists(bin_path) else 0.0
        result["status"] = "converted"
        result["ir"] = base + ".xml"
        result["detail"] = f"{base}.xml + {base}.bin ({size_mb:.1f} MB)"
        print(f"[{model_id}] ✓ Saved IR: {result['detail']}")
    except Exception as e:
        result["status"] = "save_failed"
        result["detail"] = str(e)
        print(f"[{model_id}] ✗ IR save failed: {e}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert sign-recognition ONNX models to OpenVINO IR."
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Task 1 only: report which models can be converted, do not write IR.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild IR even if it already exists.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Sign Recognition Models -> OpenVINO IR Converter")
    print("=" * 70)

    try:
        import openvino as ov
        from openvino import Core, get_version
    except ImportError:
        print()
        print("  Intel OpenVINO is not installed in this Python environment.")
        print("  Install it with:")
        print("      pip install openvino>=2024.0.0")
        print()
        return 1

    print(f"  OpenVINO version: {get_version()}")
    core = Core()
    print(f"  Available devices: {', '.join(core.available_devices)}")

    # ── Task 1: feasibility check ──
    print("\n" + "-" * 70)
    print("  TASK 1: Conversion feasibility check")
    print("-" * 70)
    feasibility = {}
    for m in MODELS:
        onnx_name = _load_onnx_file_name(m["dir"])
        onnx_path = os.path.join(m["dir"], onnx_name)
        if not os.path.exists(onnx_path):
            feasibility[m["id"]] = (False, f"ONNX not found: {onnx_name}")
            print(f"[{m['id']}] ✗ ONNX not found: {onnx_name}")
            continue
        ok, detail = check_convertible(core, onnx_path)
        feasibility[m["id"]] = (ok, detail)
        symbol = "✓" if ok else "✗"
        print(f"[{m['id']}] {symbol} {detail}")

    if args.check_only:
        print("\n(check-only mode — no IR written)")
        return 0

    # ── Tasks 2 & 3: convert and place IR in each folder ──
    print("\n" + "-" * 70)
    print("  TASKS 2 & 3: Convert and save IR into each model folder")
    print("-" * 70)
    start = time.time()
    results = []
    for m in MODELS:
        ok, _detail = feasibility.get(m["id"], (False, ""))
        if not ok:
            print(f"\n[{m['id']}] Skipping IR conversion (not convertible) — stays ONNX")
            results.append({"id": m["id"], "status": "incompatible", "ir": None,
                            "detail": _detail})
            continue
        results.append(convert_one(core, m["id"], m["dir"], args.force))

    elapsed = time.time() - start

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for r in results:
        fmt = "OpenVINO IR" if r["status"] in ("converted", "exists") else "ONNX (fallback)"
        print(f"  {r['id']:8s} -> {fmt:18s} [{r['status']}] {r['detail']}")
    print(f"\n  Completed in {elapsed:.1f}s")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
