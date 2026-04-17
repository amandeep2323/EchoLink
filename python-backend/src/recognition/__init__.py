"""
Recognition Package
=====================
ASL fingerspelling recognition pipeline.
MediaPipe Hands → PointNet classifier → Letter accumulation → Sentence building.

Based on: https://github.com/kevinjosethomas/sign-language-processing
"""

from .landmarker import Landmarker
from .recognizer import Recognizer
from .spell_corrector import SpellCorrector

__all__ = ["Landmarker", "Recognizer", "SpellCorrector"]
