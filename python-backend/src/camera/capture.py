"""
Camera Capture — Threaded OpenCV Video Capture
=================================================
Captures frames from a webcam in a background thread with
configurable resolution, FPS, mirroring, and frame queue.

Features:
  - Threaded capture (non-blocking)
  - Configurable resolution and FPS
  - Automatic horizontal mirroring (for natural selfie view)
  - Frame queue with max size (drops old frames if consumer is slow)
  - Camera disconnection detection (watchdog)
  - FPS tracking

Usage:
    camera = CameraCapture(camera_index=0, width=640, height=480, fps=30)
    camera.start()
    
    frame = camera.read()   # Get latest frame (numpy array or None)
    
    camera.stop()
"""

import time
import threading
from typing import Optional
from collections import deque

import cv2
import numpy as np


class CameraCapture:
    """Threaded webcam capture with frame queue and watchdog."""

    # If no frame for this many seconds, camera is considered disconnected
    WATCHDOG_TIMEOUT = 10.0

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        mirror: bool = True,
        max_queue_size: int = 2,
    ):
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._target_fps = fps
        self._mirror = mirror
        self._max_queue_size = max_queue_size

        # State
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Frame buffer (deque acts as a bounded queue, dropping oldest)
        self._frame_queue: deque[np.ndarray] = deque(maxlen=max_queue_size)

        # Stats
        self._frames_captured = 0
        self._frames_dropped = 0
        self._last_frame_time: Optional[float] = None
        self._fps: float = 0.0
        self._fps_counter = 0
        self._fps_timer: float = 0.0

        # Watchdog
        self._disconnected = False
        self._on_disconnect: Optional[callable] = None

    # ── Properties ──────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_disconnected(self) -> bool:
        return self._disconnected

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frames_captured(self) -> int:
        return self._frames_captured

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    # ── Lifecycle ───────────────────────────────

    def start(self, on_disconnect: Optional[callable] = None) -> None:
        """Start the camera capture thread."""
        if self._running:
            return

        self._on_disconnect = on_disconnect

        # Open the camera
        print(f"[Camera] Opening camera {self._camera_index} at {self._width}x{self._height} @ {self._target_fps}fps")
        self._cap = cv2.VideoCapture(self._camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera {self._camera_index}. "
                f"Check that the camera is connected and not in use."
            )

        # Set camera properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        # Read actual properties (camera may not support requested values)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        backend = self._cap.getBackendName()

        print(
            f"[Camera] Opened — actual: {actual_w}x{actual_h} @ {actual_fps:.0f}fps "
            f"(backend: {backend})"
        )

        # Update actual dimensions
        self._width = actual_w
        self._height = actual_h

        # Reset stats
        self._frames_captured = 0
        self._frames_dropped = 0
        self._fps = 0.0
        self._fps_counter = 0
        self._fps_timer = time.time()
        self._last_frame_time = time.time()
        self._disconnected = False

        # Start capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the camera capture thread and release the device."""
        if not self._running:
            return

        self._running = False

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        # Release camera
        if self._cap:
            self._cap.release()
            self._cap = None

        # Clear frame queue
        self._frame_queue.clear()

        print(
            f"[Camera] Stopped — captured: {self._frames_captured}, "
            f"dropped: {self._frames_dropped}"
        )

    def update_settings(
        self,
        camera_index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        mirror: Optional[bool] = None,
    ) -> bool:
        """
        Update capture settings. Returns True if camera needs restart.
        Only mirror can be changed without restart.
        """
        needs_restart = False

        if mirror is not None:
            self._mirror = mirror

        if camera_index is not None and camera_index != self._camera_index:
            self._camera_index = camera_index
            needs_restart = True

        if width is not None and width != self._width:
            self._width = width
            needs_restart = True

        if height is not None and height != self._height:
            self._height = height
            needs_restart = True

        if fps is not None and fps != self._target_fps:
            self._target_fps = fps
            needs_restart = True

        return needs_restart

    # ── Frame Access ────────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the queue.
        Returns None if no frame is available.
        Non-blocking.
        """
        with self._lock:
            if self._frame_queue:
                return self._frame_queue[-1]  # Always get the latest
            return None

    def read_and_clear(self) -> Optional[np.ndarray]:
        """
        Get the latest frame and clear the queue.
        Useful for pipelines that only need the most recent frame.
        """
        with self._lock:
            if self._frame_queue:
                frame = self._frame_queue[-1]
                self._frame_queue.clear()
                return frame
            return None

    # ── Capture Thread ──────────────────────────

    def _capture_loop(self) -> None:
        """Background thread: continuously reads frames from the camera."""
        frame_interval = 1.0 / max(self._target_fps, 1)

        while self._running and self._cap is not None:
            loop_start = time.time()

            try:
                ret, frame = self._cap.read()

                if not ret or frame is None:
                    # Check watchdog
                    if self._last_frame_time:
                        elapsed = time.time() - self._last_frame_time
                        if elapsed > self.WATCHDOG_TIMEOUT and not self._disconnected:
                            self._disconnected = True
                            print(f"[Camera] ⚠ Camera disconnected (no frames for {elapsed:.1f}s)")
                            if self._on_disconnect:
                                self._on_disconnect()
                    continue

                # Reset watchdog
                self._last_frame_time = time.time()
                self._disconnected = False

                # Mirror for natural selfie view
                if self._mirror:
                    frame = cv2.flip(frame, 1)

                # Add to queue (deque auto-drops oldest if full)
                with self._lock:
                    if len(self._frame_queue) >= self._max_queue_size:
                        self._frames_dropped += 1
                    self._frame_queue.append(frame)

                self._frames_captured += 1
                self._fps_counter += 1

                # FPS calculation (every 1 second)
                now = time.time()
                fps_elapsed = now - self._fps_timer
                if fps_elapsed >= 1.0:
                    self._fps = self._fps_counter / fps_elapsed
                    self._fps_counter = 0
                    self._fps_timer = now

            except Exception as e:
                print(f"[Camera] Capture error: {e}")
                time.sleep(0.1)
                continue

            # Frame rate limiting (if camera reads faster than target)
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    # ── Encoding ────────────────────────────────

    @staticmethod
    def encode_jpeg(frame: np.ndarray, quality: int = 80) -> Optional[bytes]:
        """Encode a frame as JPEG bytes."""
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            ret, buffer = cv2.imencode(".jpg", frame, encode_params)
            if ret:
                return buffer.tobytes()
        except Exception as e:
            print(f"[Camera] JPEG encode error: {e}")
        return None

    @staticmethod
    def encode_base64(frame: np.ndarray, quality: int = 80) -> Optional[str]:
        """Encode a frame as a base64 JPEG string (for WebSocket transport)."""
        import base64
        jpeg_bytes = CameraCapture.encode_jpeg(frame, quality)
        if jpeg_bytes:
            return base64.b64encode(jpeg_bytes).decode("ascii")
        return None
