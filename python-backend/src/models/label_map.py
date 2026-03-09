"""
Label Map — Sign Class Labels
================================
Loads and manages the mapping from model output indices to
human-readable sign/letter names.

Supports loading from:
  - .json file (list or dict)
  - .txt file (one label per line)
  - Default A-Z + special tokens (fallback)
"""

import json
import os
from typing import Optional


# Default ASL fingerspelling labels (A-Z + common special tokens)
DEFAULT_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "space", "delete", "nothing",
]


class LabelMap:
    """
    Manages the mapping from numeric class indices to sign/letter names.
    
    Usage:
        labels = LabelMap.load("path/to/labels.json")  
        labels = LabelMap.load("path/to/labels.txt")
        labels = LabelMap.default()
        
        name = labels.get_label(0)   # → "A"
        idx  = labels.get_index("B") # → 1
    """

    def __init__(self, labels: list[str]):
        self._labels = labels
        self._label_to_idx = {label: i for i, label in enumerate(labels)}

    # ── Factory Methods ─────────────────────────

    @classmethod
    def load(cls, path: str) -> "LabelMap":
        """
        Load labels from a file. Auto-detects format by extension.
        Falls back to default labels if file not found or invalid.
        """
        if not os.path.exists(path):
            print(f"[LabelMap] File not found: {path} — using default A-Z labels")
            return cls.default()

        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == ".json":
                return cls._load_json(path)
            elif ext in (".txt", ".csv"):
                return cls._load_txt(path)
            else:
                print(f"[LabelMap] Unknown format '{ext}' — using default A-Z labels")
                return cls.default()
        except Exception as e:
            print(f"[LabelMap] Error loading {path}: {e} — using default A-Z labels")
            return cls.default()

    @classmethod
    def default(cls) -> "LabelMap":
        """Create a default A-Z + special tokens label map."""
        return cls(list(DEFAULT_LABELS))

    @classmethod
    def from_list(cls, labels: list[str]) -> "LabelMap":
        """Create from an explicit list of labels."""
        return cls(labels)

    @classmethod
    def auto_discover(cls, model_dir: str) -> "LabelMap":
        """
        Try to find a label file in the given directory.
        Searches for: labels.json, labels.txt, label_map.json, classes.txt
        """
        candidates = [
            "labels.json", "label_map.json", "classes.json",
            "labels.txt", "classes.txt", "label_map.txt",
        ]
        for name in candidates:
            path = os.path.join(model_dir, name)
            if os.path.exists(path):
                print(f"[LabelMap] Auto-discovered: {path}")
                return cls.load(path)

        print("[LabelMap] No label file found — using default A-Z labels")
        return cls.default()

    # ── Loaders ─────────────────────────────────

    @classmethod
    def _load_json(cls, path: str) -> "LabelMap":
        """Load labels from a JSON file (list or dict)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            labels = [str(item) for item in data]
        elif isinstance(data, dict):
            # Support {0: "A", 1: "B", ...} or {"A": 0, "B": 1, ...}
            if all(isinstance(v, int) for v in data.values()):
                # {"A": 0, "B": 1, ...} → invert
                labels = [""] * len(data)
                for name, idx in data.items():
                    if 0 <= idx < len(labels):
                        labels[idx] = str(name)
            else:
                # {0: "A", 1: "B", ...} or {"0": "A", "1": "B", ...}
                max_idx = max(int(k) for k in data.keys())
                labels = [""] * (max_idx + 1)
                for idx_str, name in data.items():
                    idx = int(idx_str)
                    if 0 <= idx < len(labels):
                        labels[idx] = str(name)
        else:
            raise ValueError(f"Unexpected JSON type: {type(data)}")

        print(f"[LabelMap] Loaded {len(labels)} labels from {path}")
        return cls(labels)

    @classmethod
    def _load_txt(cls, path: str) -> "LabelMap":
        """Load labels from a text file (one per line)."""
        with open(path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]

        print(f"[LabelMap] Loaded {len(labels)} labels from {path}")
        return cls(labels)

    # ── Accessors ───────────────────────────────

    def get_label(self, index: int) -> str:
        """Get the label name for a given class index."""
        if 0 <= index < len(self._labels):
            return self._labels[index]
        return f"unknown_{index}"

    def get_index(self, label: str) -> Optional[int]:
        """Get the class index for a given label name."""
        return self._label_to_idx.get(label)

    @property
    def labels(self) -> list[str]:
        """All labels in order."""
        return list(self._labels)

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return len(self._labels)

    def __len__(self) -> int:
        return len(self._labels)

    def __repr__(self) -> str:
        preview = self._labels[:5]
        suffix = f", ... ({len(self._labels)} total)" if len(self._labels) > 5 else ""
        return f"LabelMap({preview}{suffix})"
