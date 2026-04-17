"""
Camera Package
================
Camera capture, frame compositing, and virtual camera output.
"""

from .capture import CameraCapture
from .compositor import FrameCompositor
from .virtual_camera import VirtualCamera

__all__ = ["CameraCapture", "FrameCompositor", "VirtualCamera"]
