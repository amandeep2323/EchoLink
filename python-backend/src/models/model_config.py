"""
Model Config — model.json Schema Definition
===============================================
Each model folder contains a model.json that describes everything
the pipeline needs: model file, input format, labels, preprocessing,
inference type, and post-processing parameters.

The ModelConfig dataclass validates and provides defaults for all fields.

Usage:
    config = ModelConfig.load("models/sign/model1/model.json")
    print(config.name)                    # "PointNet ASL Fingerspelling"
    print(config.input.landmark_source)   # "mediapipe_hands"
    print(config.inference.type)          # "single_frame"
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ═══════════════════════════════════════════════
# Input Configuration
# ═══════════════════════════════════════════════

@dataclass
class InputConfig:
    """How the model expects input data."""

    # Which MediaPipe model to use for landmarks
    # "mediapipe_hands" → 21 landmarks per hand (current)
    # "mediapipe_holistic" → hands + pose (future)
    landmark_source: str = "mediapipe_hands"

    # MediaPipe parameters
    max_hands: int = 1
    detection_confidence: float = 0.75
    tracking_confidence: float = 0.75
    model_complexity: int = 0

    # Model input shape (batch, landmarks, dims)
    # e.g., [1, 21, 3] for PointNet, [1, 30, 177] for LSTM
    input_shape: list[int] = field(default_factory=lambda: [1, 21, 3])

    # How many dimensions to use: 2 = (x,y), 3 = (x,y,z)
    # "auto" means the pipeline will probe the model at startup
    use_dimensions: str | int = "auto"

    # Normalization method:
    # "min_max" — normalize each axis to [0,1] independently (current)
    # "wrist_relative" — subtract wrist position, scale by shoulder width
    # "none" — pass raw landmarks
    normalize: str = "min_max"

    @classmethod
    def from_dict(cls, data: dict) -> "InputConfig":
        """Create from a dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ═══════════════════════════════════════════════
# Inference Configuration
# ═══════════════════════════════════════════════

@dataclass
class InferenceConfig:
    """How to run inference with the model."""

    # Inference type:
    # "single_frame" — classify one frame at a time (PointNet)
    # "sequence" — classify a sequence of frames (LSTM)
    type: str = "single_frame"

    # Minimum raw confidence to consider a prediction valid
    confidence_threshold: float = 0.60

    # Inference backend: "onnx" (recommended) or "keras"
    backend: str = "onnx"

    # Sequence model parameters (only used if type == "sequence")
    sequence_length: int = 30    # Number of frames per sequence
    stride: int = 5              # Frames to skip between sequences

    # Whether to apply softmax if model output isn't normalized
    apply_softmax: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "InferenceConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ═══════════════════════════════════════════════
# Post-Processing Configuration
# ═══════════════════════════════════════════════

@dataclass
class PostprocessConfig:
    """How to post-process model predictions."""

    # Whether to apply misrecognition fixes (A/T, D/I, F/W)
    misrecognition_fixes: bool = True

    # Whether to run spell correction when words are finalized
    spell_correction: bool = True

    # Letter accumulation thresholds
    stability_frames: int = 8       # Frames of same letter → accept new
    repeat_frames: int = 18         # Frames of same letter → repeat
    max_consecutive: int = 2        # Max consecutive identical letters

    # Cooldown between letter accepts
    cooldown_frames: int = 5        # Same letter cooldown
    diff_cooldown_frames: int = 4   # Different letter cooldown

    # Progressive acceptance
    high_confidence_threshold: float = 0.85  # Accept faster above this
    high_confidence_frames: int = 3           # Frames needed when high conf

    # Confidence smoothing
    smooth_window: int = 10         # Rolling average window size
    min_smooth_confidence: float = 0.70  # Minimum smoothed confidence

    # Hand activity filtering
    movement_threshold: float = 0.015  # Min movement to consider active
    movement_history: int = 5          # Frames to track movement

    # Word/sentence timing
    word_timeout: float = 1.5       # Seconds of no-hand → finalize word
    sentence_timeout: float = 5.0   # Seconds of silence → complete sentence

    @classmethod
    def from_dict(cls, data: dict) -> "PostprocessConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ═══════════════════════════════════════════════
# Main Model Configuration
# ═══════════════════════════════════════════════

@dataclass
class ModelConfig:
    """
    Complete model configuration loaded from model.json.

    Each model folder must have a model.json with at minimum:
      - name
      - model_file

    Everything else has sensible defaults matching the current PointNet model.
    """

    # ── Required ──
    name: str = "Unnamed Model"
    model_file: str = ""               # Filename within the model folder

    # ── Metadata ──
    version: str = "1.0"
    author: str = ""
    description: str = ""
    type: str = "fingerspelling"       # "fingerspelling", "word_recognition", etc.

    # ── Labels ──
    # Can be a string ("ABCDEFGHIKLMNOPQRSTUVWXY"),
    # a list (["A", "B", "C", ...]),
    # or a filename ("labels.json")
    labels: str | list[str] = "ABCDEFGHIKLMNOPQRSTUVWXY"

    # ── Sub-configs ──
    input: InputConfig = field(default_factory=InputConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    # ── Runtime (set after loading, not in JSON) ──
    config_path: str = field(default="", repr=False)    # Path to model.json
    model_dir: str = field(default="", repr=False)      # Directory containing model.json

    # ── Loading ─────────────────────────────────

    @classmethod
    def load(cls, config_path: str) -> "ModelConfig":
        """
        Load a ModelConfig from a model.json file.
        Validates required fields and provides defaults for everything else.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = cls._from_dict(data)
        config.config_path = os.path.abspath(config_path)
        config.model_dir = os.path.dirname(os.path.abspath(config_path))

        # Validate
        config._validate()

        return config

    @classmethod
    def _from_dict(cls, data: dict) -> "ModelConfig":
        """Create ModelConfig from a dictionary, parsing nested configs."""
        # Extract nested configs
        input_data = data.pop("input", {})
        inference_data = data.pop("inference", {})
        postprocess_data = data.pop("postprocess", {})

        # Filter top-level to only known fields
        top_fields = {
            "name", "model_file", "version", "author",
            "description", "type", "labels",
        }
        top_data = {k: v for k, v in data.items() if k in top_fields}

        return cls(
            **top_data,
            input=InputConfig.from_dict(input_data),
            inference=InferenceConfig.from_dict(inference_data),
            postprocess=PostprocessConfig.from_dict(postprocess_data),
        )

    def _validate(self) -> None:
        """Validate the configuration."""
        errors = []

        # model_file is required
        if not self.model_file:
            errors.append("'model_file' is required in model.json")

        # Check model_file exists (if model_dir is set)
        if self.model_dir and self.model_file:
            model_path = os.path.join(self.model_dir, self.model_file)
            if not os.path.exists(model_path):
                errors.append(
                    f"Model file not found: {model_path}\n"
                    f"  Expected '{self.model_file}' in {self.model_dir}"
                )

        # Validate landmark source
        valid_sources = {"mediapipe_hands", "mediapipe_holistic"}
        if self.input.landmark_source not in valid_sources:
            errors.append(
                f"Invalid landmark_source: '{self.input.landmark_source}'. "
                f"Must be one of: {valid_sources}"
            )

        # Validate inference type
        valid_types = {"single_frame", "sequence"}
        if self.inference.type not in valid_types:
            errors.append(
                f"Invalid inference type: '{self.inference.type}'. "
                f"Must be one of: {valid_types}"
            )

        # Validate normalization
        valid_norms = {"min_max", "wrist_relative", "none"}
        if self.input.normalize not in valid_norms:
            errors.append(
                f"Invalid normalize: '{self.input.normalize}'. "
                f"Must be one of: {valid_norms}"
            )

        if errors:
            error_str = "\n  ".join(errors)
            raise ValueError(
                f"Invalid model.json ({self.config_path}):\n  {error_str}"
            )

    # ── Convenience ─────────────────────────────

    @property
    def model_path(self) -> str:
        """Full path to the model weights file."""
        if self.model_dir and self.model_file:
            return os.path.join(self.model_dir, self.model_file)
        return self.model_file

    @property
    def labels_list(self) -> list[str]:
        """Get labels as a list of strings."""
        if isinstance(self.labels, list):
            return self.labels
        if isinstance(self.labels, str):
            # Check if it's a filename
            if self.labels.endswith(".json") or self.labels.endswith(".txt"):
                label_path = os.path.join(self.model_dir, self.labels)
                if os.path.exists(label_path):
                    with open(label_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        return [str(x) for x in data]
                    return list(data.keys()) if isinstance(data, dict) else list(self.labels)
            # It's a character string like "ABCDEF..."
            return list(self.labels)
        return []

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return len(self.labels_list)

    @property
    def model_id(self) -> str:
        """
        Model ID derived from the folder name.
        e.g., "models/sign/model1/model.json" → "model1"
        """
        if self.model_dir:
            return os.path.basename(self.model_dir)
        return "unknown"

    def to_dict(self) -> dict:
        """Serialize to a dictionary (for JSON export)."""
        d = asdict(self)
        # Remove runtime fields
        d.pop("config_path", None)
        d.pop("model_dir", None)
        return d

    def to_info(self) -> dict:
        """Return a summary dict for the frontend."""
        return {
            "id": self.model_id,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "type": self.type,
            "num_classes": self.num_classes,
            "labels_preview": "".join(self.labels_list[:10]) + ("..." if self.num_classes > 10 else ""),
            "model_file": self.model_file,
            "landmark_source": self.input.landmark_source,
            "inference_type": self.inference.type,
            "input_shape": self.input.input_shape,
        }

    def __repr__(self) -> str:
        return (
            f"ModelConfig(id='{self.model_id}', name='{self.name}', "
            f"type='{self.type}', classes={self.num_classes}, "
            f"file='{self.model_file}')"
        )
