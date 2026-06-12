"""
Recognizer — ASL Fingerspelling Recognition Pipeline
================================================================
Full pipeline: model loading → classification → misrecognition fixes
→ confidence smoothing → hand stability check → letter accumulation
→ spell correction → word/sentence building.

Phase 6: All thresholds are config-driven via model.json.
Falls back to sensible defaults if no config is provided.

Based on: https://github.com/kevinjosethomas/sign-language-processing
"""

import os
import time
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from collections import deque

from ..models import ModelLoader, LabelMap


# ═══════════════════════════════════════════════
# Default Constants (used when no config provided)
# ═══════════════════════════════════════════════

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"

DEFAULTS = {
    "stability_frames": 5,
    "repeat_frames": 15,
    "cooldown_frames": 5,
    "diff_cooldown_frames": 2,
    "max_consecutive_same": 2,
    "confidence_smooth_window": 10,
    "min_confidence": 0.60,
    "min_smooth_confidence": 0.70,
    "high_confidence_threshold": 0.85,
    "high_confidence_window": 3,
    "movement_threshold": 0.015,
    "movement_history": 5,
    "word_timeout_seconds": 1.5,
    "sentence_timeout_seconds": 5.0,
    "spell_correction": True,
    "misrecognition_fixes": True,
}


# ═══════════════════════════════════════════════
# Recognition Result
# ═══════════════════════════════════════════════

@dataclass
class RecognitionResult:
    """Result from a single frame's recognition processing."""

    # Detection
    letter: str = ""
    confidence: float = 0.0
    smoothed_confidence: float = 0.0
    top_3: list = field(default_factory=list)
    hands_detected: bool = False

    # Letter accumulation
    letter_added: bool = False
    added_letter: str = ""
    rejection_reason: str = ""

    # Transcript (SEPARATED: completed vs current)
    current_word: str = ""
    completed_text: str = ""
    full_transcript: str = ""
    transcript_changed: bool = False
    is_sentence_complete: bool = False

    # Quality indicators
    hand_movement: float = 0.0
    hand_active: bool = False


# ═══════════════════════════════════════════════
# Recognizer
# ═══════════════════════════════════════════════

class Recognizer:
    """
    ASL fingerspelling recognizer.
    All thresholds are config-driven via ModelConfig (Phase 6).
    """

    def __init__(
        self,
        model_path: str = "",
        model_dir: str = "",
        min_confidence: float = 0.60,
        use_gpu: bool = False,
    ):
        self._model_path = model_path
        self._model_dir = model_dir
        self._use_gpu = use_gpu

        # Model
        self._loader = ModelLoader()
        self._labels = LabelMap.from_list(list(LETTERS))
        self._model_input_dim: int = 3
        self._config = None  # ModelConfig reference

        # ── Config-driven thresholds (set from config or defaults) ──
        self._stability_frames = DEFAULTS["stability_frames"]
        self._repeat_frames = DEFAULTS["repeat_frames"]
        self._cooldown_frames = DEFAULTS["cooldown_frames"]
        self._diff_cooldown_frames = DEFAULTS["diff_cooldown_frames"]
        self._max_consecutive_same = DEFAULTS["max_consecutive_same"]
        self._confidence_smooth_window = DEFAULTS["confidence_smooth_window"]
        self._min_confidence = min_confidence
        self._min_smooth_confidence = DEFAULTS["min_smooth_confidence"]
        self._high_confidence_threshold = DEFAULTS["high_confidence_threshold"]
        self._high_confidence_window = DEFAULTS["high_confidence_window"]
        self._movement_threshold = DEFAULTS["movement_threshold"]
        self._movement_history_size = DEFAULTS["movement_history"]
        self._word_timeout = DEFAULTS["word_timeout_seconds"]
        self._sentence_timeout = DEFAULTS["sentence_timeout_seconds"]
        self._use_spell_correction = DEFAULTS["spell_correction"]
        self._use_misrecognition_fixes = DEFAULTS["misrecognition_fixes"]

        # ── Letter accumulation state ──
        self._raw_letters: list[str] = []
        self._raw_word: str = ""
        self._words: list[str] = []
        self._corrected_words: list[str] = []
        self._previous_transcript: str = ""

        # ── Confidence smoothing ──
        self._confidence_history: deque[tuple[str, float]] = deque(
            maxlen=self._confidence_smooth_window
        )

        # ── Hand movement tracking ──
        self._prev_landmarks: Optional[np.ndarray] = None
        self._movement_history: deque[float] = deque(
            maxlen=self._movement_history_size
        )

        # ── Letter cooldown ──
        self._frames_since_last_accept: int = 999
        self._last_accepted_letter: str = ""

        # ── Sequence buffer (for sequence-based models) ──
        self._sequence_buffer: deque = deque()

        # ── Timing ──
        self._last_hand_time: float = 0.0
        self._last_letter_time: float = 0.0
        self._sentence_complete: bool = False
        self._word_finalized_this_gap: bool = False

        # ── Spell corrector ──
        self._spell_corrector = None
        if self._use_spell_correction:
            try:
                from .spell_corrector import SpellCorrector
                self._spell_corrector = SpellCorrector()
                print("[Recognizer] ✓ Spell corrector loaded")
            except Exception as e:
                print(f"[Recognizer] ⚠ Spell corrector not available: {e}")

    # ── Properties ──────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loader.is_loaded

    @property
    def min_confidence(self) -> float:
        return self._min_confidence

    @min_confidence.setter
    def min_confidence(self, value: float) -> None:
        self._min_confidence = max(0.0, min(1.0, value))

    @property
    def current_word(self) -> str:
        return self._raw_word

    @property
    def completed_text(self) -> str:
        return " ".join(self._corrected_words).strip()

    @property
    def completed_words(self) -> list[str]:
        return list(self._corrected_words)

    @property
    def full_transcript(self) -> str:
        parts = list(self._corrected_words)
        if self._raw_word:
            parts.append(self._raw_word)
        return " ".join(parts).strip()

    @property
    def model_info(self) -> dict:
        return self._loader.model_info

    @property
    def config(self):
        return self._config

    # ── Apply Config ────────────────────────────

    def _apply_config(self, config) -> None:
        """Apply thresholds from a ModelConfig object."""
        self._config = config

        pp = config.postprocess

        self._stability_frames = pp.stability_frames
        self._repeat_frames = pp.repeat_frames
        self._cooldown_frames = pp.cooldown_frames
        self._diff_cooldown_frames = pp.diff_cooldown_frames
        self._max_consecutive_same = pp.max_consecutive
        self._confidence_smooth_window = pp.smooth_window
        self._min_smooth_confidence = pp.min_smooth_confidence
        self._high_confidence_threshold = pp.high_confidence_threshold
        self._high_confidence_window = pp.high_confidence_frames
        self._word_timeout = pp.word_timeout
        self._sentence_timeout = pp.sentence_timeout
        self._use_spell_correction = pp.spell_correction
        self._use_misrecognition_fixes = pp.misrecognition_fixes

        # Use config min_confidence if not overridden
        self._min_confidence = config.inference.confidence_threshold

        # Resize deques with new window sizes
        self._confidence_history = deque(
            self._confidence_history,
            maxlen=self._confidence_smooth_window
        )
        self._movement_history = deque(
            self._movement_history,
            maxlen=self._movement_history_size
        )

        # Resize sequence buffer for sequence-based models
        seq_len = getattr(config.inference, 'sequence_length', 50)
        self._sequence_buffer = deque(
            self._sequence_buffer,
            maxlen=seq_len
        )

        # Load/unload spell corrector based on config
        if self._use_spell_correction and self._spell_corrector is None:
            try:
                from .spell_corrector import SpellCorrector
                self._spell_corrector = SpellCorrector()
                print("[Recognizer] ✓ Spell corrector loaded (from config)")
            except Exception as e:
                print(f"[Recognizer] ⚠ Spell corrector not available: {e}")
        elif not self._use_spell_correction:
            self._spell_corrector = None

        print(f"[Recognizer] Config applied:")
        print(f"  stability_frames={self._stability_frames}, "
              f"repeat_frames={self._repeat_frames}")
        print(f"  cooldown={self._cooldown_frames}, "
              f"diff_cooldown={self._diff_cooldown_frames}")
        print(f"  min_confidence={self._min_confidence:.2f}, "
              f"min_smooth={self._min_smooth_confidence:.2f}")
        print(f"  high_conf={self._high_confidence_threshold:.2f} "
              f"(window={self._high_confidence_window})")
        print(f"  spell_correction={self._use_spell_correction}, "
              f"misrecognition_fixes={self._use_misrecognition_fixes}")

    # ── Loading ─────────────────────────────────

    def load(self, model_dir: str = "", config=None) -> None:
        """
        Load the sign language model.

        Args:
            model_dir: Directory containing model files.
                       If empty, uses self._model_dir.
            config: Optional ModelConfig. If provided, thresholds and
                    model loading use config values.
        """
        if model_dir:
            self._model_dir = model_dir

        # ── Apply config if provided ──
        if config is not None:
            self._apply_config(config)
            # Use config-driven loading
            self._loader.load_from_config(config)
        else:
            # Try to find model.json in directory for auto-config
            model_json = os.path.join(self._model_dir, "model.json")
            if os.path.isfile(model_json):
                try:
                    from ..models.model_config import ModelConfig
                    auto_config = ModelConfig.load(model_json)
                    self._apply_config(auto_config)
                    self._loader.load_from_config(auto_config)
                    print(f"[Recognizer] Auto-loaded config from {model_json}")
                except Exception as e:
                    print(f"[Recognizer] ⚠ Failed to load model.json: {e}")
                    self._load_legacy()
            else:
                self._load_legacy()

        # Update labels from loader
        if self._loader.is_loaded and self._loader._labels:
            self._labels = self._loader._labels

        # Auto-detect input dimensions
        self._detect_input_dims()

        letters_str = "".join(self._labels.labels) if self._labels else LETTERS
        print(
            f"[Recognizer] ✓ Ready — "
            f"letters: {letters_str} ({len(self._labels.labels) if self._labels else 24} classes), "
            f"input_dim: {self._model_input_dim}, "
            f"min_confidence: {self._min_confidence:.2f}"
        )

    def _load_legacy(self) -> None:
        """Legacy loading: find model file in directory."""
        path = self._model_path
        if not path and self._model_dir:
            from ..models.converter import ModelConverter
            found = ModelConverter.find_model_file(self._model_dir)
            if found:
                path = found

        if not path:
            raise FileNotFoundError(
                f"No model file specified or found.\n"
                f"Set model_path or model_dir when creating Recognizer.\n"
                f"Supported formats: .onnx, .h5, .keras, .tflite"
            )

        self._loader.load(
            model_path=path,
            labels=self._labels,
            use_gpu=self._use_gpu,
        )

    def _detect_input_dims(self) -> None:
        """Auto-detect whether model needs 2 or 3 input dimensions."""
        if self._config and self._config.input.use_dimensions:
            self._model_input_dim = self._config.input.use_dimensions
            if self._model_input_dim == "auto":
                self._model_input_dim = 3
            print(
                f"[Recognizer] Input dims set from config: "
                f"{self._model_input_dim}"
            )
            return

        # Probe model
        test_3d = self._test_input_dims(3)
        test_2d = self._test_input_dims(2)

        if test_3d and not test_2d:
            self._model_input_dim = 3
            print("[Recognizer] Model requires 3 dims (x,y,z) — Conv1D/PointNet")
        elif test_2d and test_3d:
            self._model_input_dim = 2
            print("[Recognizer] Model accepts both dims — using 2 (x,y)")
        elif test_2d:
            self._model_input_dim = 2
            print("[Recognizer] Model requires 2 dims (x,y)")
        else:
            self._model_input_dim = 3
            print("[Recognizer] ⚠ Could not determine dims — defaulting to 3")

    def _test_input_dims(self, dims: int) -> bool:
        try:
            dummy = np.random.rand(1, 21, dims).astype(np.float32)
            self._loader.predict_raw(dummy)
            return True
        except Exception:
            return False

    # ── Main Processing ─────────────────────────

    def process(
        self,
        points: Optional[np.ndarray],
        handedness: Optional[str],
        hands_detected: bool,
    ) -> RecognitionResult:
        # Branch based on inference type
        inference_type = "single_frame"
        if self._config is not None:
            inference_type = getattr(self._config.inference, 'type', 'single_frame')

        if inference_type == "sequence":
            return self._process_sequence(points, handedness, hands_detected)

        return self._process_single_frame(points, handedness, hands_detected)

    # ── Single-Frame Processing (PointNet / fingerspelling) ──

    def _process_single_frame(
        self,
        points: Optional[np.ndarray],
        handedness: Optional[str],
        hands_detected: bool,
    ) -> RecognitionResult:
        result = RecognitionResult(hands_detected=hands_detected)
        now = time.time()

        self._frames_since_last_accept += 1

        if hands_detected and points is not None:
            self._last_hand_time = now
            self._sentence_complete = False
            self._word_finalized_this_gap = False

            # ── Step 1: Track hand movement ──
            movement = self._compute_movement(points)
            result.hand_movement = movement
            result.hand_active = movement > self._movement_threshold

            # ── Step 2: Classify ──
            letter, confidence, top_3 = self._classify(points)
            result.letter = letter
            result.confidence = confidence
            result.top_3 = top_3

            # ── Step 3: Fix misrecognition ──
            if self._use_misrecognition_fixes:
                letter = self._fix_misrecognition(
                    letter, points, handedness or "right"
                )
                result.letter = letter

            # ── Step 4: Smooth confidence ──
            self._confidence_history.append((letter, confidence))
            smoothed = self._get_smoothed_confidence(letter)
            result.smoothed_confidence = smoothed

            # ── Step 5: Record raw letter ──
            self._raw_letters.append(letter)

            # ── Step 6: Accumulate (with quality gates) ──
            if smoothed > self._min_confidence:
                added, reason = self._try_add_letter(
                    letter, smoothed, movement
                )
                result.letter_added = added
                result.rejection_reason = reason
                if added:
                    result.added_letter = letter
                    self._last_letter_time = now
                    self._frames_since_last_accept = 0
                    self._last_accepted_letter = letter
            else:
                result.rejection_reason = (
                    f"low smoothed confidence "
                    f"({smoothed:.2f} < {self._min_confidence:.2f})"
                )

        else:
            # ── No hand detected ──
            self._confidence_history.clear()
            self._prev_landmarks = None
            self._movement_history.clear()

            # Finalize current word after timeout
            if (
                self._raw_word
                and not self._word_finalized_this_gap
                and self._last_hand_time > 0
                and (now - self._last_hand_time) > self._word_timeout
            ):
                self._finalize_word()
                self._word_finalized_this_gap = True

            # Sentence timeout
            if self._last_hand_time > 0:
                silence = now - self._last_hand_time
                if (
                    silence > self._sentence_timeout
                    and not self._sentence_complete
                ):
                    result.is_sentence_complete = True
                    self._sentence_complete = True

        # ── Build transcript ──
        result.current_word = self._raw_word
        result.completed_text = self.completed_text
        result.full_transcript = self.full_transcript

        if result.full_transcript != self._previous_transcript:
            result.transcript_changed = True
            self._previous_transcript = result.full_transcript

        return result

    # ── Sequence Processing (WLASL / word-level) ──

    def _process_sequence(
        self,
        points: Optional[np.ndarray],
        handedness: Optional[str],
        hands_detected: bool,
    ) -> RecognitionResult:
        """Process using sequence buffering for word-level models."""
        result = RecognitionResult(hands_detected=hands_detected)
        now = time.time()

        self._frames_since_last_accept += 1

        seq_len = self._sequence_buffer.maxlen or 50

        if hands_detected and points is not None:
            self._last_hand_time = now
            self._sentence_complete = False
            self._word_finalized_this_gap = False

            # Append frame to sequence buffer
            frame_points = np.array(points, dtype=np.float32)
            
            # Handle different point formats
            if frame_points.ndim == 1:
                # Already flat (e.g., aggregate_3d mode: [3])
                pass
            elif frame_points.ndim == 2:
                # 2D array (e.g., [55, 2] or [1, 3])
                if frame_points.shape[0] == 1:
                    # Squeeze batch dimension: [1, 3] -> [3]
                    frame_points = frame_points.squeeze(0)
                # else keep as is for normal keypoint format [55, 2]
            
            self._sequence_buffer.append(frame_points)

            # Only classify when buffer is full
            if len(self._sequence_buffer) == seq_len:
                # Build tensor based on model's tensor_format
                stacked = np.stack(list(self._sequence_buffer), axis=0)
                tensor_format = (
                    getattr(self._config.inference, "tensor_format", "nodes_first")
                    if self._config
                    else "nodes_first"
                )
                
                # Check if we're using aggregate features (1D per frame) or keypoints (2D per frame)
                if stacked.ndim == 2:
                    # Aggregate features: stacked shape is [frames, features]
                    # e.g., [543, 3] for Model3
                    tensor = np.expand_dims(stacked, axis=0).astype(np.float32)
                else:
                    # Normal keypoints: stacked shape is [frames, nodes, coords]
                    num_nodes = stacked.shape[1]
                    num_coords = stacked.shape[2]

                    if tensor_format == "frames_nodes_coords":
                        # Format: [frames, nodes, coords] (raw sequence, no batch dim)
                        # Used by Model3 ONNX exported from TFLite signature.
                        tensor = stacked.astype(np.float32)
                    elif tensor_format == "frames_first":
                        # Format: [batch, frames, nodes*coords]
                        # stacked shape: [frames, nodes, coords]
                        tensor = stacked.reshape(seq_len, num_nodes * num_coords)
                        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
                    else:
                        # Format: [batch, nodes, frames*coords] (default, for Model2)
                        # stacked shape: [frames, nodes, coords]
                        tensor = stacked.transpose(1, 0, 2).reshape(
                            num_nodes, seq_len * num_coords
                        )
                        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

                # Classify
                word, confidence, top_3 = self._classify_raw(tensor)
                result.letter = word
                result.confidence = confidence
                result.top_3 = top_3

                # Confidence smoothing
                self._confidence_history.append((word, confidence))
                smoothed = self._get_smoothed_confidence(word)
                result.smoothed_confidence = smoothed

                # Gate: smoothed confidence must exceed threshold
                if smoothed < self._min_smooth_confidence:
                    # Let deque auto-rotate: pop one oldest frame so next
                    # frame advances the window by 1
                    self._sequence_buffer.popleft()
                    result.rejection_reason = (
                        f"low confidence ({smoothed:.3f} < {self._min_smooth_confidence:.2f})"
                    )
                else:
                    # Cooldown: same word needs cooldown_frames elapsed
                    time_since_last = (
                        now - self._last_letter_time
                        if self._last_letter_time > 0
                        else float('inf')
                    )
                    cooldown_sec = self._cooldown_frames / 10.0
                    diff_cooldown_sec = self._diff_cooldown_frames / 10.0

                    is_same = word == self._last_accepted_letter
                    cd = cooldown_sec if is_same else diff_cooldown_sec

                    if time_since_last < cd:
                        self._sequence_buffer.popleft()
                        result.rejection_reason = (
                            f"cooldown ({time_since_last:.1f}s / {cd:.1f}s)"
                        )
                    else:
                        # ✅ Accept the word
                        print(
                            f"[Recognizer] ACCEPT: '{word}' "
                            f"(raw={confidence:.3f}, smooth={smoothed:.3f})"
                        )
                        result.letter_added = True
                        result.added_letter = word
                        self._corrected_words.append(word)
                        self._last_letter_time = now
                        self._last_accepted_letter = word
                        self._frames_since_last_accept = 0
                        # Clear buffer after accepting
                        self._sequence_buffer.clear()
                        self._confidence_history.clear()
            else:
                result.rejection_reason = (
                    f"buffering ({len(self._sequence_buffer)}/{seq_len} frames)"
                )
        else:
            # No hands — don't clear buffer (allow brief dropouts)
            self._confidence_history.clear()

            # Sentence timeout
            if self._last_hand_time > 0:
                silence = now - self._last_hand_time
                if (
                    silence > self._sentence_timeout
                    and not self._sentence_complete
                ):
                    result.is_sentence_complete = True
                    self._sentence_complete = True

        # Build transcript
        result.current_word = ""
        result.completed_text = self.completed_text
        result.full_transcript = self.full_transcript

        if result.full_transcript != self._previous_transcript:
            result.transcript_changed = True
            self._previous_transcript = result.full_transcript

        return result

    # ── Classification ──────────────────────────

    def _classify(
        self, points: np.ndarray
    ) -> tuple[str, float, list[dict]]:
        """Classify single-frame landmarks (PointNet / fingerspelling)."""
        if not self._loader.is_loaded:
            return "", 0.0, []

        features = points[:, :, :self._model_input_dim].astype(np.float32)
        sign, confidence, top_3 = self._loader.predict_sign(
            features, top_k=3
        )
        return sign, confidence, top_3

    def _classify_raw(
        self, tensor: np.ndarray
    ) -> tuple[str, float, list[dict]]:
        """Classify a pre-formatted tensor (sequence models)."""
        if not self._loader.is_loaded:
            return "", 0.0, []

        sign, confidence, top_3 = self._loader.predict_sign(
            tensor, top_k=3
        )
        return sign, confidence, top_3

    # ── Confidence Smoothing ────────────────────

    def _get_smoothed_confidence(self, letter: str) -> float:
        if not self._confidence_history:
            return 0.0

        letter_confs = [
            conf for (l, conf) in self._confidence_history if l == letter
        ]
        if not letter_confs:
            return 0.0

        return sum(letter_confs) / len(letter_confs)

    # ── Hand Movement Tracking ──────────────────

    def _compute_movement(self, points: np.ndarray) -> float:
        current = points[0]  # (21, 3)

        if self._prev_landmarks is None:
            self._prev_landmarks = current.copy()
            return 0.0

        diffs = current[:, :2] - self._prev_landmarks[:, :2]
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        avg_movement = float(np.mean(distances))

        self._prev_landmarks = current.copy()
        self._movement_history.append(avg_movement)

        return avg_movement

    # ── Misrecognition Fixes ────────────────────

    @staticmethod
    def _fix_misrecognition(
        letter: str, points: np.ndarray, handedness: str,
    ) -> str:
        pts = points[0]  # (21, 3)

        # A vs T
        if letter in ("A", "T"):
            thumb_tip = pts[4]
            thumb_middle = pts[3]
            index_tip = pts[8]

            if handedness == "left":
                if (
                    thumb_tip[0] > index_tip[0]
                    and thumb_middle[0] > index_tip[0]
                ):
                    letter = "A"
                else:
                    letter = "T"
            else:
                if (
                    thumb_tip[0] < index_tip[0]
                    and thumb_middle[0] < index_tip[0]
                ):
                    letter = "A"
                else:
                    letter = "T"

        # D vs I
        if letter in ("D", "I"):
            index_tip = pts[8]
            pinky_tip = pts[20]
            if index_tip[1] > pinky_tip[1]:
                letter = "I"
            else:
                letter = "D"

        # F vs W
        if letter in ("F", "W"):
            index_tip = pts[8]
            pinky_tip = pts[20]
            if index_tip[1] > pinky_tip[1]:
                letter = "F"
            else:
                letter = "W"

        return letter

    # ── Letter Accumulation ─────────────────────

    def _try_add_letter(
        self, letter: str, confidence: float, movement: float
    ) -> tuple[bool, str]:
        # ── Gate 1: Letter cooldown ──
        if letter == self._last_accepted_letter:
            if self._frames_since_last_accept < self._cooldown_frames:
                return False, (
                    f"cooldown ({self._frames_since_last_accept}/"
                    f"{self._cooldown_frames} frames)"
                )
        else:
            if self._frames_since_last_accept < self._diff_cooldown_frames:
                return False, (
                    f"diff-letter cooldown "
                    f"({self._frames_since_last_accept}/"
                    f"{self._diff_cooldown_frames} frames)"
                )

        # ── Gate 2: Progressive acceptance window ──
        if confidence >= self._high_confidence_threshold:
            effective_window = self._high_confidence_window
        else:
            effective_window = self._stability_frames

        effective_repeat_window = self._repeat_frames

        added = False
        reason = ""

        # Path 1: Repeat a letter (requires sustained hold)
        if len(self._raw_letters) >= effective_repeat_window:
            last_n = set(self._raw_letters[-effective_repeat_window:])
            if len(last_n) == 1:
                if (
                    len(self._raw_word) < self._max_consecutive_same
                    or self._raw_word[-self._max_consecutive_same:]
                    != letter * self._max_consecutive_same
                ):
                    self._raw_word += letter
                    added = True

        # Path 2: Add a new/different letter
        if not added and len(self._raw_letters) >= effective_window:
            last_n = set(self._raw_letters[-effective_window:])
            if len(last_n) == 1:
                if not self._raw_word or self._raw_word[-1] != letter:
                    self._raw_word += letter
                    added = True
                else:
                    reason = "duplicate (word already ends with this letter)"
            else:
                unique_in_window = len(last_n)
                reason = (
                    f"unstable (last {effective_window} frames have "
                    f"{unique_in_window} different letters: "
                    f"{', '.join(sorted(last_n))})"
                )

        if not added and not reason:
            reason = (
                f"insufficient frames ({len(self._raw_letters)}/"
                f"{effective_window})"
            )

        return added, reason

    # ── Word Finalization ───────────────────────

    def _finalize_word(self) -> None:
        raw = self._raw_word.strip()
        if not raw:
            return

        self._words.append(raw)

        # Spell correction
        corrected = raw
        if (
            self._use_spell_correction
            and self._spell_corrector
            and len(raw) >= 2
        ):
            corrected_word, was_corrected, ratio = (
                self._spell_corrector.correct_with_info(raw)
            )
            if was_corrected:
                print(
                    f"[Recognizer] Spell correction: '{raw}' → "
                    f"'{corrected_word}' (similarity: {ratio:.2f})"
                )
                corrected = corrected_word

        self._corrected_words.append(corrected)

        # Reset word-building state
        self._raw_word = ""
        self._raw_letters.clear()
        self._frames_since_last_accept = 999
        self._last_accepted_letter = ""

    # ── State Management ────────────────────────

    def reset(self) -> None:
        self._raw_letters.clear()
        self._raw_word = ""
        self._words.clear()
        self._corrected_words.clear()
        self._previous_transcript = ""
        self._last_hand_time = 0.0
        self._last_letter_time = 0.0
        self._sentence_complete = False
        self._word_finalized_this_gap = False
        self._confidence_history.clear()
        self._prev_landmarks = None
        self._movement_history.clear()
        self._frames_since_last_accept = 999
        self._last_accepted_letter = ""
        self._sequence_buffer.clear()
        print("[Recognizer] State reset")

    def clear_transcript(self) -> None:
        self._words.clear()
        self._corrected_words.clear()
        self._raw_word = ""
        self._raw_letters.clear()
        self._previous_transcript = ""
        self._frames_since_last_accept = 999
        self._last_accepted_letter = ""
        self._sequence_buffer.clear()
        print("[Recognizer] Transcript cleared")

    def get_latest_word(self) -> str:
        if self._corrected_words:
            return self._corrected_words[-1]
        return ""

    def get_completed_text(self) -> str:
        return " ".join(self._corrected_words).strip()

    # ── Cleanup ─────────────────────────────────

    def release(self) -> None:
        self._loader.unload()
        self.reset()
        print("[Recognizer] Released")

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
