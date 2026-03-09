"""
Models Package
================
Auto-detection, loading, and registry of sign language models.
Supports .onnx, .h5, .keras, and .tflite formats.

Multi-model support via ModelRegistry:
  - Each model lives in its own subfolder with a model.json config
  - Registry auto-discovers models, tracks active selection
  - Pipeline loads the active model seamlessly
"""

from .model_loader import ModelLoader
from .label_map import LabelMap
from .model_config import ModelConfig, InputConfig, InferenceConfig, PostprocessConfig
from .model_registry import ModelRegistry

__all__ = [
    "ModelLoader",
    "LabelMap",
    "ModelConfig",
    "InputConfig",
    "InferenceConfig",
    "PostprocessConfig",
    "ModelRegistry",
]
