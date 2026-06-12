"""
Landmarker — MediaPipe/OpenPose Landmark Extraction (Phase 6: Config-Driven)
====================================================================
Extracts and normalizes hand landmarks from camera frames.

Supports config-driven settings via ModelConfig:
    - landmark_source: "mediapipe_hands", "mediapipe_holistic", or "openpose"
  - normalize: "min_max", "wrist_relative", or "none"
  - model_complexity, detection/tracking confidence, max_hands

Falls back to sensible defaults if no config is provided.

Based on: https://github.com/kevinjosethomas/sign-language-processing
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import struct
import subprocess
import cv2
import numpy as np
from typing import Optional


def _diagnose_mediapipe():
    """Print detailed diagnostic info about the mediapipe installation."""
    print("\n" + "=" * 60)
    print("  MediaPipe Diagnostic Report")
    print("=" * 60)
    print(f"  Python version: {sys.version}")
    print(f"  Python executable: {sys.executable}")

    for p in sys.path[:5]:
        shadow = os.path.join(p, "mediapipe.py")
        if os.path.exists(shadow):
            print(f"  ⚠ SHADOWING DETECTED: {shadow}")
            print(f"    Delete or rename this file.")

    try:
        import mediapipe
        mp_version = getattr(mediapipe, '__version__', 'unknown')
        print(f"  mediapipe version: {mp_version}")
        print(f"  mediapipe location: {getattr(mediapipe, '__file__', 'unknown')}")

        if hasattr(mediapipe, 'solutions'):
            print("  ✓ mediapipe.solutions exists")
        else:
            print("  ✗ mediapipe.solutions NOT FOUND")

        try:
            from mediapipe.python import solutions
            print("  ✓ mediapipe.python.solutions exists")
        except ImportError as e:
            print(f"  ✗ mediapipe.python.solutions: {e}")

    except ImportError as e:
        print(f"  ✗ mediapipe not installed: {e}")

    print()
    print("  FIX: pip uninstall mediapipe -y")
    print("       pip install mediapipe==0.10.14")
    print("=" * 60 + "\n")


def _load_mediapipe_holistic():
    """Load MediaPipe Holistic module. Returns (holistic_module, drawing_utils)."""

    # Strategy 1: Standard import
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'holistic'):
            holistic_mod = mp.solutions.holistic
            drawing = getattr(mp.solutions, 'drawing_utils', None)
            print("[Landmarker] \u2713 Loaded Holistic via mp.solutions.holistic")
            return holistic_mod, drawing
    except (ImportError, AttributeError) as e:
        print(f"[Landmarker] Holistic Strategy 1 failed: {e}")

    # Strategy 2: Direct submodule
    try:
        from mediapipe.python.solutions import holistic as holistic_mod
        drawing = None
        try:
            from mediapipe.python.solutions import drawing_utils as drawing
        except ImportError:
            pass
        print("[Landmarker] \u2713 Loaded Holistic via mediapipe.python.solutions.holistic")
        return holistic_mod, drawing
    except (ImportError, AttributeError) as e:
        print(f"[Landmarker] Holistic Strategy 2 failed: {e}")

    _diagnose_mediapipe()
    raise ImportError(
        "Could not load MediaPipe Holistic.\n"
        "Try: pip install mediapipe==0.10.14 --force-reinstall"
    )


def _load_mediapipe_hands():
    """Load MediaPipe Hands module. Returns (hands_module, drawing_utils, connections)."""

    # Strategy 1: Standard import
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
            hands_mod = mp.solutions.hands
            drawing = getattr(mp.solutions, 'drawing_utils', None)
            connections = hands_mod.HAND_CONNECTIONS
            print("[Landmarker] ✓ Loaded via mp.solutions.hands")
            return hands_mod, drawing, connections
    except (ImportError, AttributeError) as e:
        print(f"[Landmarker] Strategy 1 failed: {e}")

    # Strategy 2: Direct submodule
    try:
        from mediapipe.python.solutions import hands as hands_mod
        connections = hands_mod.HAND_CONNECTIONS
        drawing = None
        try:
            from mediapipe.python.solutions import drawing_utils as drawing
        except ImportError:
            pass
        print("[Landmarker] ✓ Loaded via mediapipe.python.solutions.hands")
        return hands_mod, drawing, connections
    except (ImportError, AttributeError) as e:
        print(f"[Landmarker] Strategy 2 failed: {e}")

    # Strategy 3: Direct class import
    try:
        from mediapipe.python.solutions.hands import Hands, HAND_CONNECTIONS

        class _HandsModule:
            Hands = Hands
            HAND_CONNECTIONS = HAND_CONNECTIONS

        drawing = None
        try:
            from mediapipe.python.solutions import drawing_utils as drawing
        except ImportError:
            pass
        print("[Landmarker] ✓ Loaded via direct Hands class import")
        return _HandsModule, drawing, HAND_CONNECTIONS
    except (ImportError, AttributeError) as e:
        print(f"[Landmarker] Strategy 3 failed: {e}")

    _diagnose_mediapipe()
    raise ImportError(
        "Could not load MediaPipe Hands.\n"
        "Try: pip install mediapipe==0.10.14 --force-reinstall"
    )


def _find_openpose_dir() -> str:
    """
    Locate OpenPose from env vars or the repo-local tools folder.

    The Windows OpenPose archive used by this project lives at:
      tools/openpose/openpose/
    """
    configured = os.environ.get("OPENPOSE_DIR", "").strip()

    backend_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    project_root = os.path.abspath(os.path.join(backend_root, ".."))

    candidates = [
        configured,
        os.path.join(project_root, "tools", "openpose", "openpose"),
        os.path.join(backend_root, "tools", "openpose", "openpose"),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        if (
            os.path.isdir(candidate)
            and os.path.isdir(os.path.join(candidate, "bin"))
            and os.path.isdir(os.path.join(candidate, "models"))
        ):
            return candidate

    return configured


def _configure_openpose_paths(openpose_dir: str) -> None:
    """Add OpenPose DLL and Python binding folders for Windows builds."""
    if not openpose_dir:
        return

    os.environ.setdefault("OPENPOSE_DIR", openpose_dir)

    python_paths = [
        os.path.join(openpose_dir, "python"),
        os.path.join(openpose_dir, "bin", "python"),
        os.path.join(openpose_dir, "bin", "python", "openpose", "Release"),
        os.path.join(openpose_dir, "bin", "python", "openpose", "Debug"),
    ]
    for python_path in python_paths:
        if os.path.isdir(python_path) and python_path not in sys.path:
            sys.path.append(python_path)

    if os.name == "nt":
        dll_dirs = [
            os.path.join(openpose_dir, "bin"),
            os.path.join(openpose_dir, "x64", "Release"),
        ]
        existing_path = os.environ.get("PATH", "")
        for dll_dir in dll_dirs:
            if not os.path.isdir(dll_dir):
                continue
            if dll_dir not in existing_path.split(os.pathsep):
                os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass


def _load_openpose():
    """
    Load OpenPose Python bindings.

    Uses OPENPOSE_DIR to locate the OpenPose install. Expected layout:
      OPENPOSE_DIR/
        - bin/
        - python/
        - models/
    """
    openpose_dir = _find_openpose_dir()
    _configure_openpose_paths(openpose_dir)

    try:
        from openpose import pyopenpose as op
        print("[Landmarker] Loaded OpenPose via openpose.pyopenpose")
        return op
    except ImportError:
        try:
            import pyopenpose as op
            print("[Landmarker] Loaded OpenPose via pyopenpose")
            return op
        except ImportError as e:
            raise ImportError(
                "OpenPose not available. Set OPENPOSE_DIR to the OpenPose install "
                "(with bin/python and models folders). Searched: "
                f"{openpose_dir or '[not found]'}. Python: {sys.version.split()[0]}."
            ) from e


class Landmarker:
    """
    MediaPipe/OpenPose landmark extractor.
    Phase 6: Config-driven settings via init_from_config().
    """

    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.75,
        min_tracking_confidence: float = 0.75,
        max_num_hands: int = 1,
    ):
        self._model_complexity = model_complexity
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._max_num_hands = max_num_hands
        self._normalize_mode = "min_max"
        self._landmark_source = "mediapipe_hands"
        self._active_config = None  # Store active config for feature extraction

        # OpenPose settings (lazy init)
        self._openpose = None
        self._openpose_module = None
        self._openpose_worker = None
        self._openpose_model_folder = ""
        self._openpose_net_resolution = "-1x368"
        self._openpose_hand = True
        self._openpose_face = False

        # MediaPipe objects (lazy init)
        self._hands = None
        self._holistic = None
        self._mp_drawing = None
        self._hand_connections = None
        self._initialized = False

    # ── Config-Driven Init ──────────────────────

    def init_from_config(self, config) -> None:
        """
        Apply settings from a ModelConfig object.

        Reads from config.input:
          - landmark_source: "mediapipe_hands", "mediapipe_holistic", or "openpose"
          - model_complexity: 0 or 1
          - min_detection_confidence: 0.0 - 1.0
          - min_tracking_confidence: 0.0 - 1.0
          - max_hands: 1 or 2
          - normalize: "min_max", "wrist_relative", or "none"
        """
        inp = config.input

        self._landmark_source = inp.landmark_source
        self._active_config = config  # Store for feature extraction
        self._model_complexity = inp.model_complexity
        self._min_detection_confidence = inp.detection_confidence
        self._min_tracking_confidence = inp.tracking_confidence
        self._max_num_hands = inp.max_hands
        self._normalize_mode = inp.normalize

        if self._landmark_source == "openpose":
            self._openpose_model_folder = getattr(inp, "openpose_model_folder", "")
            self._openpose_net_resolution = getattr(inp, "openpose_net_resolution", "-1x368")
            self._openpose_hand = bool(getattr(inp, "openpose_hand", True))
            self._openpose_face = bool(getattr(inp, "openpose_face", False))

        # Release existing if reconfiguring
        if self._initialized:
            self.release()

        print(f"[Landmarker] Config applied:")
        print(f"  source={self._landmark_source}, "
              f"complexity={self._model_complexity}")
        print(f"  det_conf={self._min_detection_confidence}, "
              f"track_conf={self._min_tracking_confidence}")
        print(f"  max_hands={self._max_num_hands}, "
              f"normalize={self._normalize_mode}")
        if self._landmark_source == "openpose":
            print(
                f"  openpose_model_folder={self._openpose_model_folder or '[auto]'}, "
                f"net_resolution={self._openpose_net_resolution}"
            )

    # ── Initialization ──────────────────────────

    def _ensure_initialized(self) -> None:
        """Lazy-initialize MediaPipe (Hands or Holistic based on config)."""
        if self._initialized:
            return

        if self._landmark_source == "openpose":
            openpose_import_error = None
            try:
                op = _load_openpose()
            except ImportError as e:
                op = None
                openpose_import_error = e

            model_folder = self._openpose_model_folder
            if not model_folder:
                env_model_folder = os.environ.get("OPENPOSE_MODEL_FOLDER", "").strip()
                if env_model_folder:
                    model_folder = env_model_folder

            if not model_folder:
                openpose_dir = _find_openpose_dir()
                if openpose_dir:
                    model_folder = os.path.join(openpose_dir, "models")

            if not model_folder or not os.path.isdir(model_folder):
                raise ImportError(
                    "OpenPose models folder not found. Set OPENPOSE_DIR or "
                    "openpose_model_folder in model.json."
                )

            if op is None:
                print(f"[Landmarker] Native OpenPose unavailable: {openpose_import_error}")
                self._start_openpose_worker(model_folder)
                self._initialized = True
                print(
                    f"[Landmarker] Initialized (OpenPose worker) â€” "
                    f"model_folder={model_folder}, "
                    f"net_resolution={self._openpose_net_resolution}, "
                    f"normalize={self._normalize_mode}"
                )
                return

            params = {
                "model_pose": "BODY_25",
                "model_folder": model_folder,
                "hand": self._openpose_hand,
                "face": self._openpose_face,
                "render_pose": 0,
                "display": 0,
            }
            if self._openpose_net_resolution:
                params["net_resolution"] = self._openpose_net_resolution

            wrapper = op.WrapperPython()
            wrapper.configure(params)
            wrapper.start()

            self._openpose_module = op
            self._openpose = wrapper
            self._initialized = True
            print(
                f"[Landmarker] Initialized (OpenPose) — "
                f"model_folder={model_folder}, "
                f"net_resolution={self._openpose_net_resolution}, "
                f"normalize={self._normalize_mode}"
            )
            return

        if self._landmark_source == "mediapipe_holistic":
            holistic_mod, drawing = _load_mediapipe_holistic()
            self._mp_drawing = drawing

            self._holistic = holistic_mod.Holistic(
                static_image_mode=False,
                model_complexity=self._model_complexity,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )

            self._initialized = True
            print(
                f"[Landmarker] Initialized (Holistic) — "
                f"complexity={self._model_complexity}, "
                f"det_conf={self._min_detection_confidence}, "
                f"track_conf={self._min_tracking_confidence}, "
                f"normalize={self._normalize_mode}"
            )
            return

        # Default: mediapipe_hands
        hands_mod, drawing, connections = _load_mediapipe_hands()

        self._hand_connections = connections
        self._mp_drawing = drawing

        self._hands = hands_mod.Hands(
            static_image_mode=False,
            model_complexity=self._model_complexity,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            max_num_hands=self._max_num_hands,
        )

        self._initialized = True
        print(
            f"[Landmarker] Initialized (Hands) — "
            f"complexity={self._model_complexity}, "
            f"det_conf={self._min_detection_confidence}, "
            f"track_conf={self._min_tracking_confidence}, "
            f"normalize={self._normalize_mode}"
        )

    # ── Main Processing ─────────────────────────

    def process(
        self,
        frame: np.ndarray,
        draw_landmarks: bool = True,
    ) -> tuple[
        bool, np.ndarray, Optional[np.ndarray],
        Optional[tuple], Optional[str]
    ]:
        """
        Process a BGR frame and extract landmarks.

        Returns:
            (success, annotated_frame, points, wrist_position, handedness)

        For mediapipe_hands: points shape is (1, 21, 3).
        For mediapipe_holistic: points shape is (55, 2) — 55 keypoints × (x, y).
        For openpose: points shape is (55, 2) — BODY_25 + hands mapped to WLASL order.
        """
        self._ensure_initialized()

        if self._landmark_source == "openpose":
            return self._process_openpose(frame, draw_landmarks)

        # Convert BGR → RGB
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._landmark_source == "mediapipe_holistic":
            return self._process_holistic(frame, rgb_frame, draw_landmarks)

        return self._process_hands(frame, rgb_frame, draw_landmarks)

    # ── OpenPose Processing (WLASL 55-point) ──

    def _start_openpose_worker(self, model_folder: str) -> None:
        """Start the Python 3.7 OpenPose worker used by modern Python runtimes."""
        py_launcher = shutil.which("py")
        if not py_launcher:
            raise ImportError(
                "OpenPose requires Python 3.7 bindings, but the Windows 'py' "
                "launcher was not found."
            )

        worker_path = os.path.join(os.path.dirname(__file__), "openpose_worker.py")
        if not os.path.exists(worker_path):
            raise ImportError(f"OpenPose worker not found: {worker_path}")

        openpose_dir = _find_openpose_dir()
        env = os.environ.copy()
        env["OPENPOSE_DIR"] = openpose_dir
        env["OPENPOSE_MODEL_FOLDER"] = model_folder
        env["OPENPOSE_NET_RESOLUTION"] = self._openpose_net_resolution
        env["OPENPOSE_HAND"] = "1" if self._openpose_hand else "0"
        env["OPENPOSE_FACE"] = "1" if self._openpose_face else "0"

        self._openpose_worker = subprocess.Popen(
            [py_launcher, "-3.7", "-u", worker_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        )

        ready = self._read_worker_message()
        if ready.get("status") != "ready":
            self._stop_openpose_worker()
            raise ImportError(f"OpenPose worker failed to start: {ready}")

    def _read_worker_exact(self, size: int) -> bytes:
        if not self._openpose_worker or not self._openpose_worker.stdout:
            raise RuntimeError("OpenPose worker is not running")

        chunks = []
        remaining = size
        while remaining > 0:
            chunk = self._openpose_worker.stdout.read(remaining)
            if not chunk:
                raise RuntimeError("OpenPose worker stopped unexpectedly")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_worker_message(self) -> dict:
        header = self._read_worker_exact(4)
        size = struct.unpack("<I", header)[0]
        payload = self._read_worker_exact(size)
        return json.loads(payload.decode("utf-8"))

    def _send_worker_frame(self, frame: np.ndarray) -> dict:
        if not self._openpose_worker or not self._openpose_worker.stdin:
            raise RuntimeError("OpenPose worker is not running")

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Could not encode frame for OpenPose worker")

        payload = encoded.tobytes()
        self._openpose_worker.stdin.write(struct.pack("<I", len(payload)))
        self._openpose_worker.stdin.write(payload)
        self._openpose_worker.stdin.flush()
        return self._read_worker_message()

    def _stop_openpose_worker(self) -> None:
        worker = self._openpose_worker
        self._openpose_worker = None
        if not worker:
            return
        try:
            if worker.stdin:
                worker.stdin.write(struct.pack("<I", 0))
                worker.stdin.flush()
        except Exception:
            pass
        try:
            worker.terminate()
            worker.wait(timeout=2)
        except Exception:
            try:
                worker.kill()
            except Exception:
                pass

    def _process_openpose(
        self,
        frame: np.ndarray,
        draw_landmarks: bool,
    ) -> tuple[
        bool, np.ndarray, Optional[np.ndarray],
        Optional[tuple], Optional[str]
    ]:
        """
        Process using OpenPose BODY_25 + hands and map to 55 keypoints.

        Returns points as numpy array of shape (55, 2).
        """
        if self._openpose_worker is not None:
            return self._process_openpose_worker(frame, draw_landmarks)

        frame.flags.writeable = True

        datum = self._openpose_module.Datum()
        datum.cvInputData = frame
        try:
            self._openpose.emplaceAndPop([datum])
        except TypeError:
            self._openpose.emplaceAndPop(self._openpose_module.VectorDatum([datum]))

        pose_keypoints = getattr(datum, "poseKeypoints", None)
        hand_keypoints = getattr(datum, "handKeypoints", None)

        left_hand = None
        right_hand = None
        if hand_keypoints is not None and len(hand_keypoints) >= 2:
            left_hand = hand_keypoints[0]
            right_hand = hand_keypoints[1]

        if pose_keypoints is None and left_hand is None and right_hand is None:
            return False, frame, None, None, None

        raw_points = self._extract_openpose_55_points(
            pose_keypoints, left_hand, right_hand
        )
        if raw_points is None:
            return False, frame, None, None, None

        h, w = frame.shape[:2]
        points = self._normalize_openpose_points(raw_points, w, h)

        annotated = frame
        if draw_landmarks:
            annotated = self._draw_openpose_points(frame, raw_points)

        wrist_pos = None
        right_wrist = points[4]
        left_wrist = points[7]
        if np.any(right_wrist):
            wrist_pos = (float(right_wrist[0]), float(right_wrist[1]))
        elif np.any(left_wrist):
            wrist_pos = (float(left_wrist[0]), float(left_wrist[1]))

        return True, annotated, points, wrist_pos, None

    def _process_openpose_worker(
        self,
        frame: np.ndarray,
        draw_landmarks: bool,
    ) -> tuple[
        bool, np.ndarray, Optional[np.ndarray],
        Optional[tuple], Optional[str]
    ]:
        """Process a frame through the Python 3.7 OpenPose worker."""
        frame.flags.writeable = True
        result = self._send_worker_frame(frame)
        if "error" in result:
            raise RuntimeError(f"OpenPose worker error: {result['error']}")

        pose_keypoints = (
            np.array(result["pose"], dtype=np.float32)
            if result.get("pose") is not None
            else None
        )
        left_hand = (
            np.array(result["left_hand"], dtype=np.float32)
            if result.get("left_hand") is not None
            else None
        )
        right_hand = (
            np.array(result["right_hand"], dtype=np.float32)
            if result.get("right_hand") is not None
            else None
        )

        if pose_keypoints is None and left_hand is None and right_hand is None:
            return False, frame, None, None, None

        raw_points = self._extract_openpose_55_points(
            pose_keypoints, left_hand, right_hand
        )
        if raw_points is None:
            return False, frame, None, None, None

        h, w = frame.shape[:2]
        points = self._normalize_openpose_points(raw_points, w, h)

        annotated = frame
        if draw_landmarks:
            annotated = self._draw_openpose_points(frame, raw_points)

        wrist_pos = None
        right_wrist = points[4]
        left_wrist = points[7]
        if np.any(right_wrist):
            wrist_pos = (float(right_wrist[0]), float(right_wrist[1]))
        elif np.any(left_wrist):
            wrist_pos = (float(left_wrist[0]), float(left_wrist[1]))

        return True, annotated, points, wrist_pos, None

    # ── Hands Processing (existing) ─────────────

    def _process_hands(
        self,
        frame: np.ndarray,
        rgb_frame: np.ndarray,
        draw_landmarks: bool,
    ) -> tuple[
        bool, np.ndarray, Optional[np.ndarray],
        Optional[tuple], Optional[str]
    ]:
        """Process using MediaPipe Hands (single-hand, 21 landmarks)."""
        results = self._hands.process(rgb_frame)
        frame.flags.writeable = True

        if not results.multi_hand_landmarks:
            return False, frame, None, None, None

        # Draw landmarks on frame
        if draw_landmarks and self._mp_drawing and self._hand_connections:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_spec = self._mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=8, circle_radius=8
                )
                connection_spec = self._mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=6, circle_radius=2
                )
                self._mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self._hand_connections,
                    landmark_spec,
                    connection_spec,
                )

        # Extract first hand's landmarks
        hand = results.multi_hand_landmarks[0]
        raw_points = np.array(
            [(lm.x, lm.y, lm.z) for lm in hand.landmark]
        )

        # Normalize
        points = self._normalize_points(raw_points)

        # Apply feature extraction mode if specified
        feature_mode = getattr(self._active_config.input, 'feature_mode', 'full') if hasattr(self, '_active_config') and self._active_config else 'full'
        if feature_mode == "aggregate_3d":
            points = self._extract_aggregate_3d_features(raw_points)

        # Wrist position for UI overlay
        wrist_pos = (hand.landmark[0].x, hand.landmark[0].y)

        # Handedness
        handedness = "right"
        if results.multi_handedness:
            try:
                handedness = (
                    results.multi_handedness[0]
                    .classification[0]
                    .label.lower()
                )
            except (IndexError, AttributeError):
                pass

        return True, frame, points, wrist_pos, handedness

    # ── Holistic Processing (WLASL 55-point) ───

    def _process_holistic(
        self,
        frame: np.ndarray,
        rgb_frame: np.ndarray,
        draw_landmarks: bool,
    ) -> tuple[
        bool, np.ndarray, Optional[np.ndarray],
        Optional[tuple], Optional[str]
    ]:
        """
        Process using MediaPipe Holistic — extract 55 upper-body keypoints
        matching the WLASL OpenPose format:
          - 13 upper-body/face points (11 pose landmarks + neck + mid-hip placeholder)
          - 21 left-hand points
          - 21 right-hand points
        Returns points as numpy array of shape (55, 2).
        """
        results = self._holistic.process(rgb_frame)
        frame.flags.writeable = True

        # Check if we have any usable landmarks at all
        has_pose = results.pose_landmarks is not None
        has_left = results.left_hand_landmarks is not None
        has_right = results.right_hand_landmarks is not None

        if not has_pose and not has_left and not has_right:
            return False, frame, None, None, None

        # Draw landmarks on frame
        if draw_landmarks and self._mp_drawing:
            try:
                import mediapipe as mp
                if has_pose:
                    self._mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks,
                        mp.solutions.holistic.POSE_CONNECTIONS,
                        self._mp_drawing.DrawingSpec(
                            color=(255, 100, 0), thickness=2, circle_radius=2
                        ),
                        self._mp_drawing.DrawingSpec(
                            color=(255, 255, 255), thickness=2
                        ),
                    )
                if has_left:
                    self._mp_drawing.draw_landmarks(
                        frame, results.left_hand_landmarks,
                        mp.solutions.holistic.HAND_CONNECTIONS,
                        self._mp_drawing.DrawingSpec(
                            color=(200, 0, 200), thickness=2, circle_radius=2
                        ),
                        self._mp_drawing.DrawingSpec(
                            color=(200, 0, 200), thickness=2
                        ),
                    )
                if has_right:
                    self._mp_drawing.draw_landmarks(
                        frame, results.right_hand_landmarks,
                        mp.solutions.holistic.HAND_CONNECTIONS,
                        self._mp_drawing.DrawingSpec(
                            color=(0, 200, 0), thickness=2, circle_radius=2
                        ),
                        self._mp_drawing.DrawingSpec(
                            color=(0, 200, 0), thickness=2
                        ),
                    )
            except Exception:
                pass  # Drawing is optional

        # Feature extraction mode controls Holistic output layout.
        feature_mode = (
            getattr(self._active_config.input, "feature_mode", "full")
            if self._active_config
            else "full"
        )

        if feature_mode == "holistic_543x3":
            # Model3-compatible full holistic tensor: [543, 3]
            points = self._extract_holistic_543x3_points(results)
        else:
            # Default compact representation used by WLASL-style models.
            h, w = frame.shape[:2]
            points = self._extract_wlasl_55_points(results, w, h)

            # Optional aggregate mode for specialized compact models.
            if feature_mode == "aggregate_3d":
                points = self._extract_aggregate_3d_features(points)

        # Wrist position for UI (use pose left wrist if available)
        wrist_pos = None
        if has_pose:
            pose_lm = results.pose_landmarks.landmark
            wrist_pos = (pose_lm[15].x, pose_lm[15].y)

        return True, frame, points, wrist_pos, None

    def _extract_openpose_55_points(
        self,
        pose_keypoints: Optional[np.ndarray],
        left_hand: Optional[np.ndarray],
        right_hand: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Extract 55 points from OpenPose BODY_25 + hand keypoints.

        OpenPose keypoints format is [people, points, 3] where last dim is (x, y, score).
        """
        points_55 = np.zeros((55, 2), dtype=np.float32)

        if left_hand is not None and left_hand.ndim == 3:
            left_hand = left_hand[0] if len(left_hand) > 0 else None
        if right_hand is not None and right_hand.ndim == 3:
            right_hand = right_hand[0] if len(right_hand) > 0 else None

        # ── 1. BODY POINTS (13 points mapped to OpenPose order) ──
        if pose_keypoints is not None and len(pose_keypoints) > 0:
            person = pose_keypoints[0]

            def get_pt(idx: int) -> list[float]:
                if idx < person.shape[0] and person[idx][2] > 0.05:
                    return [float(person[idx][0]), float(person[idx][1])]
                return [0.0, 0.0]

            points_55[0] = get_pt(0)   # Nose
            points_55[1] = get_pt(1)   # Neck
            points_55[2] = get_pt(2)   # R-Shoulder
            points_55[3] = get_pt(3)   # R-Elbow
            points_55[4] = get_pt(4)   # R-Wrist
            points_55[5] = get_pt(5)   # L-Shoulder
            points_55[6] = get_pt(6)   # L-Elbow
            points_55[7] = get_pt(7)   # L-Wrist
            points_55[8] = get_pt(8)   # Mid-Hip
            points_55[9] = get_pt(15)  # R-Eye
            points_55[10] = get_pt(16) # L-Eye
            points_55[11] = get_pt(17) # R-Ear
            points_55[12] = get_pt(18) # L-Ear

        # ── 2. LEFT HAND (21 points) ──
        if left_hand is not None and len(left_hand) > 0:
            if not hasattr(self, "_last_left_hand"):
                self._last_left_hand = np.zeros((21, 2), dtype=np.float32)
            for i in range(min(21, left_hand.shape[0])):
                if left_hand[i][2] > 0.05:
                    points_55[13 + i] = [left_hand[i][0], left_hand[i][1]]
                    self._last_left_hand[i] = points_55[13 + i]
        elif hasattr(self, "_last_left_hand"):
            self._last_left_hand *= 0.95
            points_55[13:34] = self._last_left_hand

        # ── 3. RIGHT HAND (21 points) ──
        if right_hand is not None and len(right_hand) > 0:
            if not hasattr(self, "_last_right_hand"):
                self._last_right_hand = np.zeros((21, 2), dtype=np.float32)
            for i in range(min(21, right_hand.shape[0])):
                if right_hand[i][2] > 0.05:
                    points_55[34 + i] = [right_hand[i][0], right_hand[i][1]]
                    self._last_right_hand[i] = points_55[34 + i]
        elif hasattr(self, "_last_right_hand"):
            self._last_right_hand *= 0.95
            points_55[34:55] = self._last_right_hand

        return points_55

    def _extract_wlasl_55_points(self, results, frame_width: int, frame_height: int) -> np.ndarray:
        """
        Extracts exactly 55 points for the WLASL model.
        Optimized for Zoom/Meet webcam framing (face, arms, hands).
        Missing or off-screen joints are set to [0.0, 0.0].
        """
        points_55 = np.zeros((55, 2), dtype=np.float32)

        # ── 1. BODY POINTS (13 points mapped to OpenPose order) ──
        if results.pose_landmarks:
            pl = results.pose_landmarks.landmark

            # Helper: Only use the point if it is clearly visible on camera
            def get_pt(idx):
                if pl[idx].visibility > 0.35:
                    x = np.clip(pl[idx].x, 0.0, 1.0)
                    y = np.clip(pl[idx].y, 0.0, 1.0)
                    
                    return [x, y]
                return [0.0, 0.0]

            # Calculate Neck only if BOTH shoulders are clearly visible
            if pl[11].visibility > 0.5 and pl[12].visibility > 0.5:
                neck = [
                    (pl[11].x + pl[12].x) / 2.0,
                    (pl[11].y + pl[12].y) / 2.0
                ]
            else:
                neck = [0.0, 0.0]

            # STRICT OPENPOSE ORDER (0 to 12)
            points_55[0] = get_pt(0)   # Nose
            points_55[1] = neck        # Neck (Calculated)
            points_55[2] = get_pt(12)  # R-Shoulder
            points_55[3] = get_pt(14)  # R-Elbow
            points_55[4] = get_pt(16)  # R-Wrist
            points_55[5] = get_pt(11)  # L-Shoulder
            points_55[6] = get_pt(13)  # L-Elbow
            points_55[7] = get_pt(15)  # L-Wrist
            points_55[8] = [0.0, 0.0]  # Mid-Hip (ALWAYS OFF-SCREEN IN WEBCAM)
            points_55[9] = get_pt(5)   # R-Eye
            points_55[10] = get_pt(2)  # L-Eye
            points_55[11] = get_pt(8)  # R-Ear
            points_55[12] = get_pt(7)  # L-Ear

        # ── 2. LEFT HAND (21 points) ──
        if results.left_hand_landmarks:
            if not hasattr(self, '_last_left_hand'):
                self._last_left_hand = np.zeros((21, 2), dtype=np.float32)
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                points_55[13 + i] = [lm.x, lm.y]
                self._last_left_hand[i] = [lm.x, lm.y]
        elif hasattr(self, '_last_left_hand'):
            self._last_left_hand *= 0.95
            points_55[13:34] = self._last_left_hand

        # ── 3. RIGHT HAND (21 points) ──
        if results.right_hand_landmarks:
            if not hasattr(self, '_last_right_hand'):
                self._last_right_hand = np.zeros((21, 2), dtype=np.float32)
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                points_55[34 + i] = [lm.x, lm.y]
                self._last_right_hand[i] = [lm.x, lm.y]
        elif hasattr(self, '_last_right_hand'):
            self._last_right_hand *= 0.95
            points_55[34:55] = self._last_right_hand

        return points_55

    @staticmethod
    def _draw_openpose_points(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Draw simple OpenPose keypoints on the frame for debugging."""
        annotated = frame.copy()
        for x, y in points:
            if x > 0.0 and y > 0.0:
                cv2.circle(annotated, (int(x), int(y)), 2, (0, 255, 0), -1)
        return annotated

    def _normalize_openpose_points(
        self,
        points: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """Normalize OpenPose points based on configured mode."""
        if self._normalize_mode == "frame":
            points = points.copy()
            if frame_width > 0:
                points[:, 0] = points[:, 0] / float(frame_width)
            if frame_height > 0:
                points[:, 1] = points[:, 1] / float(frame_height)
            return points
        if self._normalize_mode == "min_max":
            return self._normalize_min_max_no_batch(points)
        if self._normalize_mode == "wrist_relative":
            return np.squeeze(self._normalize_wrist_relative(points), axis=0)
        return points

    @staticmethod
    def _normalize_min_max_no_batch(points: np.ndarray) -> np.ndarray:
        """Min-max normalize x and y independently without adding a batch dim."""
        points = points.copy()

        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        x_range = max_x - min_x
        if x_range > 0:
            points[:, 0] = (points[:, 0] - min_x) / x_range
        else:
            points[:, 0] = 0.0

        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        y_range = max_y - min_y
        if y_range > 0:
            points[:, 1] = (points[:, 1] - min_y) / y_range
        else:
            points[:, 1] = 0.0

        return points

    # ── Normalization ───────────────────────────

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks based on configured mode.

        Modes:
          "min_max" — Min-max per axis (matches original repo)
          "wrist_relative" — Wrist-centered, shoulder-scaled
          "frame" — Frame-based normalization (treated as raw for MediaPipe)
          "none" — Raw coordinates, just add batch dim
        """
        if self._normalize_mode == "min_max":
            return self._normalize_min_max(points)
        elif self._normalize_mode == "wrist_relative":
            return self._normalize_wrist_relative(points)
        elif self._normalize_mode == "frame":
            return np.expand_dims(points.copy(), axis=0)
        elif self._normalize_mode == "none":
            return np.expand_dims(points.copy(), axis=0)
        else:
            # Default to min_max
            return self._normalize_min_max(points)

    @staticmethod
    def _normalize_min_max(points: np.ndarray) -> np.ndarray:
        """
        Min-max normalize x and y independently.
        Exactly matches the original implementation.
        """
        points = points.copy()

        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        x_range = max_x - min_x
        if x_range > 0:
            points[:, 0] = (points[:, 0] - min_x) / x_range
        else:
            points[:, 0] = 0.0

        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        y_range = max_y - min_y
        if y_range > 0:
            points[:, 1] = (points[:, 1] - min_y) / y_range
        else:
            points[:, 1] = 0.0

        return np.expand_dims(points, axis=0)

    @staticmethod
    def _normalize_wrist_relative(points: np.ndarray) -> np.ndarray:
        """
        Shoulder-centered normalization for WLASL 55-point layout.
        Better for webcam robustness.
        """

        points = points.copy().astype(np.float32)

        # OpenPose mapping:
        # 2 = right shoulder
        # 5 = left shoulder

        right_shoulder = points[2]
        left_shoulder = points[5]

        # If shoulders missing, fallback
        if np.all(right_shoulder == 0) or np.all(left_shoulder == 0):
            return np.expand_dims(points, axis=0)

        # Center between shoulders
        center = (right_shoulder + left_shoulder) / 2.0

        points = points - center

        # Scale by shoulder width
        shoulder_dist = np.linalg.norm(
            right_shoulder - left_shoulder
        )

        if shoulder_dist > 1e-6:
            points = points / shoulder_dist

        return np.expand_dims(points, axis=0)

    @staticmethod
    def _extract_holistic_543x3_points(results) -> np.ndarray:
        """
        Extract full MediaPipe Holistic landmarks in the model3-compatible order.

        Output shape: [543, 3]
          - 468 face landmarks
          - 21 left-hand landmarks
          - 33 pose landmarks
          - 21 right-hand landmarks

        Missing groups are zero-filled.
        """
        face = np.zeros((468, 3), dtype=np.float32)
        left_hand = np.zeros((21, 3), dtype=np.float32)
        pose = np.zeros((33, 3), dtype=np.float32)
        right_hand = np.zeros((21, 3), dtype=np.float32)

        if results.face_landmarks:
            face = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark],
                dtype=np.float32,
            )
        if results.left_hand_landmarks:
            left_hand = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                dtype=np.float32,
            )
        if results.pose_landmarks:
            pose = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                dtype=np.float32,
            )
        if results.right_hand_landmarks:
            right_hand = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                dtype=np.float32,
            )

        return np.concatenate([face, left_hand, pose, right_hand], axis=0)

    @staticmethod
    def _extract_aggregate_3d_features(points: np.ndarray) -> np.ndarray:
        """
        Extract 3 aggregate features from MediaPipe Holistic landmarks.
        Used for Model3 which expects [batch, frames, 3] input.
        
        Features extracted:
        1. Dominant hand center X (normalized 0-1)
        2. Dominant hand center Y (normalized 0-1)
        3. Hand spread/scale (normalized measure of hand openness)
        
        Args:
            points: Raw 55-point array [55, 2] from MediaPipe Holistic
            
        Returns:
            Array of shape [1, 3] containing the 3 aggregate features
        """
        features = np.zeros(3, dtype=np.float32)
        
        # Extract hand landmarks
        # Points 13-33: Left hand (21 points)
        # Points 34-54: Right hand (21 points)
        left_hand = points[13:34]
        right_hand = points[34:55]
        
        # Determine which hand is more active (has more non-zero points)
        left_active = np.count_nonzero(np.any(left_hand != 0, axis=1))
        right_active = np.count_nonzero(np.any(right_hand != 0, axis=1))
        
        # Use the more active hand (or right hand if equal)
        if left_active > right_active:
            dominant_hand = left_hand
        else:
            dominant_hand = right_hand
        
        # Check if hand is detected
        hand_detected = np.any(dominant_hand != 0)
        
        if hand_detected:
            # Filter out zero points
            valid_points = dominant_hand[np.any(dominant_hand != 0, axis=1)]
            
            if len(valid_points) > 0:
                # Feature 1 & 2: Hand center (mean of all hand landmarks)
                hand_center = np.mean(valid_points, axis=0)
                features[0] = hand_center[0]  # X position
                features[1] = hand_center[1]  # Y position
                
                # Feature 3: Hand spread (measure of hand openness)
                # Calculate as the average distance from center to all points
                distances = np.linalg.norm(valid_points - hand_center, axis=1)
                hand_spread = np.mean(distances)
                features[2] = hand_spread
        
        # Return with batch dimension
        return np.expand_dims(features, axis=0)

    # ── Cleanup ─────────────────────────────────

    def release(self) -> None:
        if self._hands:
            try:
                self._hands.close()
            except Exception:
                pass
            self._hands = None
        if self._holistic:
            try:
                self._holistic.close()
            except Exception:
                pass
            self._holistic = None
        if self._openpose:
            try:
                self._openpose.stop()
            except Exception:
                pass
            self._openpose = None
        self._stop_openpose_worker()
        self._openpose_module = None
        self._mp_drawing = None
        self._hand_connections = None
        self._initialized = False
        print("[Landmarker] Released")

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
