"""
Model Registry — Auto-Discovery & Selection
================================================
Scans the models/sign/ directory for model folders,
each containing a model.json config file.

Tracks the active model via _active_model.txt persistence file.

Usage:
    registry = ModelRegistry("models/sign")
    models = registry.discover()           # Find all models
    active = registry.get_active_model()   # Get the active config
    registry.set_active_model("model2")    # Switch models
    info = registry.get_models_info()      # For frontend display
"""

import os
import json
from typing import Optional

from .model_config import ModelConfig


# Persistence file name (stored in the models/sign/ directory)
ACTIVE_MODEL_FILE = "_active_model.txt"


class ModelRegistry:
    """
    Discovers, validates, and manages sign language models.

    Each model lives in its own subfolder under the base directory:
        models/sign/
        ├── model1/
        │   ├── model.json    ← REQUIRED
        │   ├── model.onnx
        │   └── labels.json
        ├── model2/
        │   ├── model.json
        │   └── model.h5
        └── _active_model.txt ← Persists the selected model ID
    """

    def __init__(self, base_dir: str):
        """
        Args:
            base_dir: Path to the models/sign/ directory.
        """
        self._base_dir = os.path.abspath(base_dir)
        self._models: dict[str, ModelConfig] = {}  # id → config
        self._active_id: str = ""

    # ── Properties ──────────────────────────────

    @property
    def base_dir(self) -> str:
        return self._base_dir

    @property
    def models(self) -> dict[str, ModelConfig]:
        """All discovered models, keyed by ID."""
        return dict(self._models)

    @property
    def model_ids(self) -> list[str]:
        """All discovered model IDs, sorted."""
        return sorted(self._models.keys())

    @property
    def active_id(self) -> str:
        return self._active_id

    @property
    def active_model(self) -> Optional[ModelConfig]:
        """The currently active model config, or None."""
        return self._models.get(self._active_id)

    @property
    def count(self) -> int:
        return len(self._models)

    # ── Discovery ───────────────────────────────

    def discover(self) -> list[ModelConfig]:
        """
        Scan the base directory for model folders.

        A valid model folder must contain a model.json file.
        Also supports legacy layout (model files directly in base_dir
        without a model.json — creates a synthetic config).

        Returns:
            List of discovered ModelConfig objects.
        """
        self._models.clear()

        if not os.path.isdir(self._base_dir):
            print(f"[Registry] Models directory not found: {self._base_dir}")
            print(f"[Registry] Creating directory...")
            os.makedirs(self._base_dir, exist_ok=True)
            return []

        print(f"[Registry] Scanning: {self._base_dir}")

        # Scan subdirectories for model.json
        found_any = False
        for entry in sorted(os.listdir(self._base_dir)):
            entry_path = os.path.join(self._base_dir, entry)

            # Skip files, hidden dirs, and the persistence file
            if not os.path.isdir(entry_path):
                continue
            if entry.startswith("_") or entry.startswith("."):
                continue

            config_path = os.path.join(entry_path, "model.json")
            if os.path.exists(config_path):
                try:
                    config = ModelConfig.load(config_path)
                    self._models[config.model_id] = config
                    found_any = True
                    print(
                        f"[Registry] ✓ {config.model_id}: "
                        f"{config.name} ({config.num_classes} classes, "
                        f"{config.model_file})"
                    )
                except Exception as e:
                    print(f"[Registry] ✗ {entry}: Failed to load model.json — {e}")
            else:
                # Check if there are model files without model.json
                model_files = [
                    f for f in os.listdir(entry_path)
                    if f.endswith((".onnx", ".h5", ".keras", ".tflite"))
                ]
                if model_files:
                    print(
                        f"[Registry] ⚠ {entry}: Found model files "
                        f"({', '.join(model_files)}) but no model.json. "
                        f"Creating default config..."
                    )
                    config = self._create_default_config(entry_path, model_files[0])
                    if config:
                        self._models[config.model_id] = config
                        found_any = True

        # Legacy support: check for model files directly in base_dir
        if not found_any:
            legacy_config = self._check_legacy_layout()
            if legacy_config:
                self._models[legacy_config.model_id] = legacy_config

        # Load active model selection
        self._load_active_selection()

        print(
            f"[Registry] Found {len(self._models)} model(s), "
            f"active: {self._active_id or 'none'}"
        )

        return list(self._models.values())

    def _check_legacy_layout(self) -> Optional[ModelConfig]:
        """
        Support legacy layout where model files are directly in base_dir
        (not in a subfolder). Creates a synthetic model1/ folder and moves files.
        """
        model_extensions = (".onnx", ".h5", ".keras", ".tflite")
        model_files = [
            f for f in os.listdir(self._base_dir)
            if os.path.isfile(os.path.join(self._base_dir, f))
            and f.endswith(model_extensions)
        ]

        if not model_files:
            return None

        print(
            f"[Registry] Found legacy model files in base dir: "
            f"{', '.join(model_files)}"
        )
        print(f"[Registry] Migrating to model1/ subfolder...")

        # Create model1/ directory
        model1_dir = os.path.join(self._base_dir, "model1")
        os.makedirs(model1_dir, exist_ok=True)

        # Move model files
        for f in model_files:
            src = os.path.join(self._base_dir, f)
            dst = os.path.join(model1_dir, f)
            if not os.path.exists(dst):
                os.rename(src, dst)
                print(f"[Registry]   Moved: {f} → model1/{f}")

        # Move labels.json if present
        for label_file in ("labels.json", "labels.txt", "label_map.json"):
            src = os.path.join(self._base_dir, label_file)
            if os.path.exists(src):
                dst = os.path.join(model1_dir, label_file)
                if not os.path.exists(dst):
                    os.rename(src, dst)
                    print(f"[Registry]   Moved: {label_file} → model1/{label_file}")

        # Determine best model file (prefer .onnx)
        best_file = model_files[0]
        for f in model_files:
            if f.endswith(".onnx"):
                best_file = f
                break

        # Create model.json
        config = self._create_default_config(model1_dir, best_file)
        if config:
            print(f"[Registry] ✓ Legacy migration complete → model1/")

        return config

    def _create_default_config(
        self, model_dir: str, model_file: str
    ) -> Optional[ModelConfig]:
        """
        Create a default model.json for a folder that has model files
        but no config.
        """
        folder_name = os.path.basename(model_dir)

        config_data = {
            "name": f"Sign Language Model ({folder_name})",
            "model_file": model_file,
            "version": "1.0",
            "author": "Unknown",
            "description": f"Auto-discovered model from {folder_name}/",
            "type": "fingerspelling",
            "labels": "ABCDEFGHIKLMNOPQRSTUVWXY",
            "input": {
                "landmark_source": "mediapipe_hands",
                "max_hands": 1,
                "input_shape": [1, 21, 3],
                "use_dimensions": "auto",
                "normalize": "min_max",
            },
            "inference": {
                "type": "single_frame",
                "confidence_threshold": 0.60,
                "backend": "onnx",
            },
            "postprocess": {
                "misrecognition_fixes": True,
                "spell_correction": True,
                "stability_frames": 5,
                "repeat_frames": 15,
                "cooldown_frames": 5,
                "diff_cooldown_frames": 2,
            },
        }

        config_path = os.path.join(model_dir, "model.json")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)
            print(f"[Registry] ✓ Created default model.json in {folder_name}/")

            return ModelConfig.load(config_path)
        except Exception as e:
            print(f"[Registry] ✗ Failed to create model.json: {e}")
            return None

    # ── Active Model Selection ──────────────────

    def _load_active_selection(self) -> None:
        """Load the active model ID from persistence file."""
        persist_path = os.path.join(self._base_dir, ACTIVE_MODEL_FILE)

        if os.path.exists(persist_path):
            try:
                with open(persist_path, "r", encoding="utf-8") as f:
                    saved_id = f.read().strip()
                if saved_id in self._models:
                    self._active_id = saved_id
                    print(f"[Registry] Restored active model: {saved_id}")
                    return
                else:
                    print(
                        f"[Registry] ⚠ Saved model '{saved_id}' not found, "
                        f"selecting default"
                    )
            except Exception as e:
                print(f"[Registry] ⚠ Failed to read active model file: {e}")

        # Default to first model (sorted alphabetically)
        if self._models:
            self._active_id = sorted(self._models.keys())[0]
            self._save_active_selection()
            print(f"[Registry] Default active model: {self._active_id}")

    def _save_active_selection(self) -> None:
        """Persist the active model ID to disk."""
        persist_path = os.path.join(self._base_dir, ACTIVE_MODEL_FILE)
        try:
            with open(persist_path, "w", encoding="utf-8") as f:
                f.write(self._active_id)
        except Exception as e:
            print(f"[Registry] ⚠ Failed to save active model: {e}")

    def get_active_model(self) -> Optional[ModelConfig]:
        """
        Get the active model config.
        If no models are discovered, returns None.
        If the active model is invalid, falls back to first available.
        """
        if self._active_id and self._active_id in self._models:
            return self._models[self._active_id]

        # Fallback
        if self._models:
            self._active_id = sorted(self._models.keys())[0]
            return self._models[self._active_id]

        return None

    def set_active_model(self, model_id: str) -> ModelConfig:
        """
        Set the active model by ID.
        Persists the selection to disk.

        Args:
            model_id: The model folder name (e.g., "model1", "model2")

        Returns:
            The selected ModelConfig.

        Raises:
            KeyError: If the model ID is not found.
        """
        if model_id not in self._models:
            available = ", ".join(sorted(self._models.keys()))
            raise KeyError(
                f"Model '{model_id}' not found. "
                f"Available: {available or 'none'}"
            )

        old_id = self._active_id
        self._active_id = model_id
        self._save_active_selection()

        config = self._models[model_id]
        print(
            f"[Registry] Active model changed: "
            f"{old_id or 'none'} → {model_id} ({config.name})"
        )

        return config

    # ── Info for Frontend ───────────────────────

    def get_models_info(self) -> list[dict]:
        """
        Return a list of model info dicts for the frontend.
        Includes which model is currently active.
        """
        models_info = []
        for model_id in sorted(self._models.keys()):
            config = self._models[model_id]
            info = config.to_info()
            info["active"] = (model_id == self._active_id)
            models_info.append(info)
        return models_info

    def get_model_by_id(self, model_id: str) -> Optional[ModelConfig]:
        """Get a model config by ID, or None if not found."""
        return self._models.get(model_id)

    # ── Aliases (backwards compatibility) ──────

    def discover_models(self) -> list[ModelConfig]:
        """Alias for discover() — backwards compatibility."""
        return self.discover()

    def get_active_model_id(self) -> str:
        """Alias for active_id property — backwards compatibility."""
        return self._active_id

    # ── Refresh ─────────────────────────────────

    def refresh(self) -> list[ModelConfig]:
        """
        Re-scan the directory for models.
        Preserves the active selection if still valid.
        """
        saved_active = self._active_id
        models = self.discover()

        # Restore active if still valid
        if saved_active and saved_active in self._models:
            self._active_id = saved_active

        return models

    # ── Utility ─────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ModelRegistry(dir='{self._base_dir}', "
            f"models={len(self._models)}, "
            f"active='{self._active_id}')"
        )

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._models
