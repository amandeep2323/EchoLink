"""
Speech Package
================
Text-to-speech engine (Piper TTS) and virtual microphone output.
"""

from .tts_engine import TTSEngine
from .virtual_mic import VirtualMic

__all__ = ["TTSEngine", "VirtualMic"]
