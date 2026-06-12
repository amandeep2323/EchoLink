"""
OpenPose worker for Python 3.7 bindings.

The main backend can run on a modern Python version, but the Windows OpenPose
Python binding bundled with this install is compiled for CPython 3.7. This
worker keeps OpenPose isolated and communicates with the backend using a small
length-prefixed binary protocol over stdin/stdout.
"""

from __future__ import annotations

import json
import os
import struct
import sys


def _send(message: dict) -> None:
    payload = json.dumps(message).encode("utf-8")
    _PROTOCOL_OUT.write(struct.pack("<I", len(payload)))
    _PROTOCOL_OUT.write(payload)
    _PROTOCOL_OUT.flush()


def _read_exact(size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sys.stdin.buffer.read(remaining)
        if not chunk:
            raise EOFError("stdin closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _find_openpose_dir() -> str:
    configured = os.environ.get("OPENPOSE_DIR", "").strip()
    if configured:
        return os.path.abspath(configured)

    backend_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    project_root = os.path.abspath(os.path.join(backend_root, ".."))
    return os.path.join(project_root, "tools", "openpose", "openpose")


def _configure_paths(openpose_dir: str) -> None:
    for path in [
        os.path.join(openpose_dir, "python"),
        os.path.join(openpose_dir, "bin", "python"),
        os.path.join(openpose_dir, "bin", "python", "openpose", "Release"),
        os.path.join(openpose_dir, "bin", "python", "openpose", "Debug"),
    ]:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)

    bin_dir = os.path.join(openpose_dir, "bin")
    if os.path.isdir(bin_dir):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(bin_dir)
        except Exception:
            pass


def _load_openpose():
    try:
        from openpose import pyopenpose as op
        return op
    except ImportError:
        import pyopenpose as op
        return op


def _init_wrapper():
    openpose_dir = _find_openpose_dir()
    _configure_paths(openpose_dir)

    model_folder = os.environ.get("OPENPOSE_MODEL_FOLDER", "").strip()
    if not model_folder:
        model_folder = os.path.join(openpose_dir, "models")

    op = _load_openpose()
    params = {
        "model_pose": "BODY_25",
        "model_folder": model_folder,
        "hand": os.environ.get("OPENPOSE_HAND", "1") != "0",
        "face": os.environ.get("OPENPOSE_FACE", "0") == "1",
        "render_pose": 0,
        "display": 0,
    }
    net_resolution = os.environ.get("OPENPOSE_NET_RESOLUTION", "").strip()
    if net_resolution:
        params["net_resolution"] = net_resolution

    wrapper = op.WrapperPython()
    wrapper.configure(params)
    wrapper.start()
    return op, wrapper


def _process_frame(op, wrapper, payload: bytes) -> dict:
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Could not decode frame"}

    datum = op.Datum()
    datum.cvInputData = image
    try:
        wrapper.emplaceAndPop([datum])
    except TypeError:
        wrapper.emplaceAndPop(op.VectorDatum([datum]))

    pose = getattr(datum, "poseKeypoints", None)
    hands = getattr(datum, "handKeypoints", None)

    left_hand = None
    right_hand = None
    if hands is not None and len(hands) >= 2:
        left_hand = hands[0]
        right_hand = hands[1]

    return {
        "pose": pose.tolist() if pose is not None else None,
        "left_hand": left_hand.tolist() if left_hand is not None else None,
        "right_hand": right_hand.tolist() if right_hand is not None else None,
    }


def main() -> int:
    try:
        op, wrapper = _init_wrapper()
        _send({"status": "ready"})

        while True:
            header = sys.stdin.buffer.read(4)
            if not header:
                break
            size = struct.unpack("<I", header)[0]
            if size == 0:
                break
            payload = _read_exact(size)
            try:
                _send(_process_frame(op, wrapper, payload))
            except Exception as e:
                _send({"error": str(e)})

        try:
            wrapper.stop()
        except Exception:
            pass
        return 0
    except Exception as e:
        _send({"status": "error", "error": str(e)})
        return 1


# Preserve stdout for the protocol, then redirect noisy OpenPose logs to stderr.
_PROTOCOL_OUT = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
