"""
Landmarker — MediaPipe Landmark Extraction (Phase 6: Config-Driven)
====================================================================
Extracts and normalizes hand landmarks from camera frames.

Supports config-driven settings via ModelConfig:
  - landmark_source: "mediapipe_hands" or "mediapipe_holistic"
  - normalize: "min_max", "wrist_relative", or "none"
  - model_complexity, detection/tracking confidence, max_hands

Falls back to sensible defaults if no config is provided.

Based on: https://github.com/kevinjosethomas/sign-language-processing
"""

import os
import sys
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


class Landmarker:
    """
    MediaPipe landmark extractor.
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

        # MediaPipe objects (lazy init)
        self._hands = None
        self._mp_drawing = None
        self._hand_connections = None
        self._initialized = False

    # ── Config-Driven Init ──────────────────────

    def init_from_config(self, config) -> None:
        """
        Apply settings from a ModelConfig object.

        Reads from config.input:
          - landmark_source: "mediapipe_hands" or "mediapipe_holistic"
          - model_complexity: 0 or 1
          - min_detection_confidence: 0.0 - 1.0
          - min_tracking_confidence: 0.0 - 1.0
          - max_hands: 1 or 2
          - normalize: "min_max", "wrist_relative", or "none"
        """
        inp = config.input

        self._landmark_source = inp.landmark_source
        self._model_complexity = inp.model_complexity
        self._min_detection_confidence = inp.detection_confidence
        self._min_tracking_confidence = inp.tracking_confidence
        self._max_num_hands = inp.max_hands
        self._normalize_mode = inp.normalize

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

    # ── Initialization ──────────────────────────

    def _ensure_initialized(self) -> None:
        """Lazy-initialize MediaPipe."""
        if self._initialized:
            return

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
            f"[Landmarker] Initialized — "
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
        Process a BGR frame and extract hand landmarks.

        Returns:
            (success, annotated_frame, points, wrist_position, handedness)
        """
        self._ensure_initialized()

        # Convert BGR → RGB
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run MediaPipe
        results = self._hands.process(rgb_frame)

        frame.flags.writeable = True

        # No hand detected
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

    # ── Normalization ───────────────────────────

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks based on configured mode.

        Modes:
          "min_max" — Min-max per axis (matches original repo)
          "wrist_relative" — Wrist-centered, shoulder-scaled
          "none" — Raw coordinates, just add batch dim
        """
        if self._normalize_mode == "min_max":
            return self._normalize_min_max(points)
        elif self._normalize_mode == "wrist_relative":
            return self._normalize_wrist_relative(points)
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
        Wrist-centered normalization.
        All points are relative to wrist (landmark 0).
        Scaled by hand span (max distance from wrist).
        """
        points = points.copy()
        wrist = points[0].copy()

        # Center on wrist
        points = points - wrist

        # Scale by max distance from wrist
        distances = np.sqrt(np.sum(points ** 2, axis=1))
        max_dist = np.max(distances)
        if max_dist > 0:
            points = points / max_dist

        return np.expand_dims(points, axis=0)

    # ── Cleanup ─────────────────────────────────

    def release(self) -> None:
        if self._hands:
            try:
                self._hands.close()
            except Exception:
                pass
            self._hands = None
        self._mp_drawing = None
        self._hand_connections = None
        self._initialized = False
        print("[Landmarker] Released")

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
