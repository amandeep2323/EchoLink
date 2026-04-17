"""
Virtual Camera — OBS Virtual Camera Output
=============================================
Sends composited video frames to a virtual camera device
(OBS Virtual Camera) via pyvirtualcam. This allows other
applications like Google Meet, Zoom, and Teams to use the
SignSpeak video feed as a camera input.

Prerequisites:
  - OBS Studio installed (registers the virtual camera driver)
  - pyvirtualcam installed: pip install pyvirtualcam

Usage:
    vcam = VirtualCamera(width=640, height=480, fps=30)
    vcam.start()
    
    vcam.send(frame)   # Send a BGR numpy frame
    
    vcam.stop()
"""

import threading
from typing import Optional

import numpy as np


class VirtualCamera:
    """Outputs video frames to OBS Virtual Camera via pyvirtualcam."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self._width = width
        self._height = height
        self._fps = fps

        # State
        self._cam = None          # pyvirtualcam.Camera instance
        self._running = False
        self._lock = threading.Lock()
        self._frames_sent = 0

    # ── Properties ──────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frames_sent(self) -> int:
        return self._frames_sent

    # ── Lifecycle ───────────────────────────────

    def start(self) -> None:
        """
        Start the virtual camera output.
        
        Raises:
            RuntimeError: If pyvirtualcam is not installed or
                          OBS Virtual Camera driver is not available.
        """
        if self._running:
            return

        try:
            import pyvirtualcam
        except ImportError as e:
            raise RuntimeError(
                f"pyvirtualcam not installed: {e}\n"
                f"Install with: pip install pyvirtualcam\n"
                f"Also ensure OBS Studio is installed for the virtual camera driver."
            ) from e

        try:
            self._cam = pyvirtualcam.Camera(
                width=self._width,
                height=self._height,
                fps=self._fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
            self._running = True
            self._frames_sent = 0
            print(
                f"[VCam] Started — {self._width}x{self._height} @ {self._fps}fps "
                f"(device: {self._cam.device})"
            )
        except Exception as e:
            self._cam = None
            raise RuntimeError(
                f"Failed to start virtual camera: {e}\n"
                f"Make sure OBS Studio is installed and has been run at least once\n"
                f"to register the virtual camera driver."
            ) from e

    def stop(self) -> None:
        """Stop the virtual camera output."""
        if not self._running:
            return

        self._running = False

        with self._lock:
            if self._cam:
                try:
                    self._cam.close()
                except Exception as e:
                    print(f"[VCam] Error closing: {e}")
                finally:
                    self._cam = None

        print(f"[VCam] Stopped — {self._frames_sent} frames sent")

    # ── Frame Output ────────────────────────────

    def send(self, frame: np.ndarray) -> bool:
        """
        Send a BGR frame to the virtual camera.
        
        The frame is automatically resized if it doesn't match
        the virtual camera's resolution.
        
        Args:
            frame: BGR numpy array (h, w, 3)
            
        Returns:
            True if the frame was sent successfully, False otherwise.
        """
        if not self._running or self._cam is None:
            return False

        try:
            import cv2

            # Resize if needed
            h, w = frame.shape[:2]
            if w != self._width or h != self._height:
                frame = cv2.resize(frame, (self._width, self._height))

            # Ensure correct dtype and channels
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            with self._lock:
                if self._cam:
                    self._cam.send(frame)
                    self._cam.sleep_until_next_frame()
                    self._frames_sent += 1
                    return True

        except Exception as e:
            print(f"[VCam] Send error: {e}")

        return False

    # ── Settings Update ─────────────────────────

    def update_settings(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> bool:
        """
        Update virtual camera settings. Returns True if restart is needed.
        Settings only take effect after restart.
        """
        needs_restart = False

        if width is not None and width != self._width:
            self._width = width
            needs_restart = True

        if height is not None and height != self._height:
            self._height = height
            needs_restart = True

        if fps is not None and fps != self._fps:
            self._fps = fps
            needs_restart = True

        return needs_restart

    def __del__(self):
        self.stop()
