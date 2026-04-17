"""
Pipeline Manager
==================
Manages the full SignSpeak pipeline lifecycle.

Phase 6 Step 5:
  - ModelRegistry integration for multi-model support
  - switch_model() for hot-swapping models
  - Config-driven landmarker + recognizer initialization
  - Model info in status updates
"""

import os
import sys
import asyncio
import time
import traceback
from typing import Optional, Callable, Awaitable, List
from dataclasses import dataclass


# ═══════════════════════════════════════════════
# Directory Paths
# ═══════════════════════════════════════════════

_BACKEND_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

SIGN_MODEL_DIR = os.path.join(_BACKEND_ROOT, "models", "sign")
TTS_MODEL_DIR = os.path.join(_BACKEND_ROOT, "models", "tts")


# ═══════════════════════════════════════════════
# Pipeline Settings
# ═══════════════════════════════════════════════

@dataclass
class PipelineSettings:
    camera_index: int = 0
    resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    show_landmarks: bool = True
    show_overlay: bool = True
    tts_enabled: bool = False
    vcam_enabled: bool = False
    vcam_mirror: bool = False
    vmic_enabled: bool = False
    vmic_device: str = ""
    confidence_threshold: float = 0.6
    tts_voice: str = "en_US-lessac-medium"
    audio_output_device: str = ""
    active_model: str = ""  # Phase 6: model ID

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "PipelineSettings":
        settings = cls()
        if not data:
            return settings
        for key, value in data.items():
            if hasattr(settings, key):
                if key == "resolution" and isinstance(value, (list, tuple)):
                    value = (int(value[0]), int(value[1]))
                setattr(settings, key, value)
        return settings

    def update(self, data: Optional[dict]) -> None:
        if not data:
            return
        for key, value in data.items():
            if hasattr(self, key):
                if key == "resolution" and isinstance(value, (list, tuple)):
                    value = (int(value[0]), int(value[1]))
                setattr(self, key, value)


# ═══════════════════════════════════════════════
# Pipeline Manager
# ═══════════════════════════════════════════════

BroadcastFn = Callable[[str], Awaitable[None]]


class PipelineManager:
    MAX_CONSECUTIVE_ERRORS = 10

    def __init__(self):
        # State
        self._running: bool = False
        self._starting: bool = False
        self._settings: PipelineSettings = PipelineSettings()
        self._task: Optional[asyncio.Task] = None
        self._broadcast_fn: Optional[BroadcastFn] = None

        # Stats
        self._frames_processed: int = 0
        self._fps: float = 0.0
        self._start_time: Optional[float] = None
        self._last_fps_time: float = 0.0
        self._fps_frame_count: int = 0

        # Subsystem states
        self._model_loaded: bool = False
        self._hands_detected: bool = False
        self._vcam_active: bool = False
        self._vmic_active: bool = False

        # Error tracking
        self._consecutive_errors: int = 0
        self._last_error_msg: str = ""

        # Subsystem instances
        self._camera = None
        self._landmarker = None
        self._recognizer = None
        self._compositor = None
        self._vcam = None
        self._tts = None
        self._vmic = None

        # TTS tracking
        self._last_tts_text: str = ""
        self._last_spoken_word_count: int = 0

        # Critical flags
        self._landmarker_ok: bool = False
        self._recognizer_ok: bool = False

        # Camera recovery
        self._camera_recovery_attempts: int = 0
        self._max_camera_recovery: int = 3
        self._last_camera_recovery: float = 0.0

        # ── Phase 6: Model Registry ──
        self._registry = None
        self._active_config = None  # Current ModelConfig
        self._init_registry()

    # ── Phase 6: Registry Initialization ────────

    def _init_registry(self) -> None:
        """Initialize the model registry on construction."""
        try:
            from ..models import ModelRegistry
            self._registry = ModelRegistry(SIGN_MODEL_DIR)
            models = self._registry.discover()
            print(f"[Pipeline] Model registry: {len(models)} model(s) discovered")
            for m in models:
                active_marker = " ◄ ACTIVE" if m.model_id == self._registry.active_id else ""
                print(f"[Pipeline]   • {m.model_id}: {m.name} ({m.type}){active_marker}")
        except Exception as e:
            print(f"[Pipeline] ⚠ Model registry init failed: {e}")
            self._registry = None

    # ── Properties ──────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def settings(self) -> PipelineSettings:
        return self._settings

    @property
    def available_models(self) -> list:
        """Return list of discovered models."""
        if self._registry:
            return list(self._registry.models.values())
        return []

    @property
    def active_model_id(self) -> str:
        """Return the active model ID."""
        if self._registry:
            return self._registry.active_id
        return ""

    @property
    def active_model_name(self) -> str:
        """Return the active model display name."""
        if self._active_config:
            return self._active_config.name
        return "Unknown"

    def get_status(self) -> dict:
        return {
            "pipeline_running": self._running,
            "model_loaded": self._model_loaded,
            "hands_detected": self._hands_detected,
            "vcam_active": self._vcam_active,
            "vmic_active": self._vmic_active,
            "fps": round(self._fps, 1),
            "frames_processed": self._frames_processed,
            # Phase 6: model info
            "model_id": self.active_model_id,
            "model_name": self.active_model_name,
            "available_models": len(self.available_models),
        }

    def get_models_list(self) -> list:
        """Return list of model info dicts for the frontend."""
        if not self._registry:
            return []

        active_id = self._registry.active_id
        result = []
        for mid in self._registry.model_ids:
            m = self._registry.models[mid]
            result.append({
                "id": m.model_id,
                "name": m.name,
                "description": m.description,
                "type": m.type,
                "model_type": m.inference.type,
                "version": m.version,
                "author": m.author,
                "labels_count": m.num_classes,
                "active": m.model_id == active_id,
            })
        return result

    def set_broadcast(self, broadcast_fn: BroadcastFn) -> None:
        self._broadcast_fn = broadcast_fn

    # ── Pipeline Lifecycle ──────────────────────

    async def start(self, settings_data: Optional[dict]) -> None:
        if self._running or self._starting:
            return

        self._starting = True

        self._settings = PipelineSettings.from_dict(settings_data)
        self._running = True
        self._frames_processed = 0
        self._fps = 0.0
        self._start_time = time.time()
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        self._consecutive_errors = 0
        self._last_error_msg = ""
        self._last_tts_text = ""
        self._last_spoken_word_count = 0

        # Phase 6: Resolve which model to load
        active_id = self.active_model_id
        if self._settings.active_model and self._registry:
            # Frontend specified a model
            try:
                self._registry.set_active_model(self._settings.active_model)
                active_id = self._settings.active_model
            except KeyError:
                print(f"[Pipeline] ⚠ Requested model '{self._settings.active_model}' not found, using default")

        print(
            f"[Pipeline] Starting — camera={self._settings.camera_index}, "
            f"resolution={self._settings.resolution}, fps={self._settings.fps}"
        )
        print(f"[Pipeline] Model dir: {SIGN_MODEL_DIR}")
        print(f"[Pipeline] TTS dir:   {TTS_MODEL_DIR}")
        print(f"[Pipeline] Active model: {active_id}")

        await self._init_subsystems()

        self._starting = False

        if not self._camera:
            print("[Pipeline] ✗ Cannot start — no camera available")
            self._running = False
            await self._broadcast_error("Pipeline failed to start: camera not available")
            return

        if not self._landmarker_ok:
            print("[Pipeline] ⚠ Landmarker not available — frames will show without landmarks")

        if not self._recognizer_ok:
            print("[Pipeline] ⚠ Recognizer not available — no sign detection")

        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        elapsed = time.time() - self._start_time if self._start_time else 0
        avg_fps = self._frames_processed / elapsed if elapsed > 0 else 0
        print(
            f"[Pipeline] Stopped — "
            f"frames={self._frames_processed}, "
            f"duration={elapsed:.1f}s, "
            f"avg_fps={avg_fps:.1f}"
        )

        await self._release_subsystems()

        self._hands_detected = False
        self._vcam_active = False
        self._vmic_active = False
        self._fps = 0.0
        self._task = None
        self._landmarker_ok = False
        self._recognizer_ok = False

    def update_settings(self, data: Optional[dict]) -> None:
        if not data:
            return

        self._settings.update(data)

        if self._recognizer and "confidence_threshold" in data:
            self._recognizer.min_confidence = self._settings.confidence_threshold

        # ── Resolution/FPS change → restart camera ──
        needs_camera_restart = False
        if "resolution" in data and self._camera and self._running:
            needs_camera_restart = True
        if "fps" in data and self._camera and self._running:
            needs_camera_restart = True
        if "camera_index" in data and self._camera and self._running:
            needs_camera_restart = True

        if needs_camera_restart:
            self._restart_camera()

        # ── VCam toggle ──
        if "vcam_enabled" in data:
            if data["vcam_enabled"] and not self._vcam_active:
                self._start_vcam()
            elif not data["vcam_enabled"] and self._vcam_active:
                self._stop_vcam()

        # ── VCam mirror ──
        if "vcam_mirror" in data:
            print(f"[Pipeline] VCam mirror: {'ON' if data['vcam_mirror'] else 'OFF'}")

        # ── TTS toggle ──
        if "tts_enabled" in data:
            if data["tts_enabled"]:
                self._ensure_tts_ready()
            elif self._tts:
                self._tts.stop()
                print("[Pipeline] TTS paused (thread stopped)")

        # ── Audio output device (local speaker) ──
        if "audio_output_device" in data and self._tts:
            device_name = data["audio_output_device"] or ""
            self._tts.set_local_device(device_name)
            print(f"[Pipeline] TTS local playback → '{device_name or 'disabled'}'")

        # ── VMic toggle ──
        if "vmic_enabled" in data:
            if data["vmic_enabled"] and not self._vmic_active:
                self._start_vmic()
            elif not data["vmic_enabled"] and self._vmic_active:
                self._stop_vmic()

        print(f"[Pipeline] Settings updated: {data}")

    def clear_transcript(self) -> None:
        if self._recognizer:
            self._recognizer.reset()
        self._last_tts_text = ""
        self._last_spoken_word_count = 0
        print("[Pipeline] Transcript cleared")

    # ── Phase 6: Model Switching ────────────────

    async def switch_model(self, model_id: str) -> dict:
        """
        Switch to a different model.
        Hot-swaps if landmark source is the same (keeps camera + landmarker running).
        Full restart if landmark source differs.

        Returns a dict with result info.
        """
        if not self._registry:
            return {"success": False, "error": "Model registry not available"}

        # Validate model exists
        target = self._registry.get_model_by_id(model_id)

        if not target:
            return {"success": False, "error": f"Model '{model_id}' not found"}

        # Already active?
        if model_id == self._registry.active_id:
            return {"success": True, "message": f"Model '{model_id}' is already active"}

        print(f"[Pipeline] Switching model: {self._registry.get_active_model_id()} → {model_id}")

        # Load new config
        try:
            from ..models import ModelConfig
            new_config = ModelConfig.load(target.config_path)
        except Exception as e:
            return {"success": False, "error": f"Failed to load config: {e}"}

        # Determine if we can hot-swap
        old_landmark_source = None
        if self._active_config:
            old_landmark_source = self._active_config.input.landmark_source

        new_landmark_source = new_config.input.landmark_source
        can_hot_swap = (
            self._running
            and old_landmark_source == new_landmark_source
            and self._landmarker_ok
        )

        if can_hot_swap:
            # ── Hot-swap: keep camera + landmarker, only swap recognizer ──
            print(f"[Pipeline] Hot-swapping recognizer (same landmark source: {new_landmark_source})")

            try:
                # Update registry
                self._registry.set_active_model(model_id)
                self._active_config = new_config

                # Release old recognizer
                if self._recognizer:
                    self._recognizer.release()
                    self._recognizer = None
                    self._recognizer_ok = False
                    self._model_loaded = False

                # Load new recognizer with new config
                from ..recognition import Recognizer
                model_dir = os.path.join(SIGN_MODEL_DIR, model_id)

                self._recognizer = Recognizer(
                    model_dir=model_dir,
                    min_confidence=self._settings.confidence_threshold,
                    use_gpu=False,
                )
                self._recognizer.load(config=new_config)
                self._model_loaded = True
                self._recognizer_ok = True

                # Update landmarker config (thresholds may have changed)
                if self._landmarker:
                    self._landmarker.init_from_config(new_config)

                # Reset TTS word tracking
                self._last_spoken_word_count = 0
                self._last_tts_text = ""

                print(f"[Pipeline] ✓ Hot-swapped to '{new_config.name}'")

                # Broadcast updated status
                if self._broadcast_fn:
                    await self._broadcast_status()

                return {
                    "success": True,
                    "message": f"Switched to '{new_config.name}' (hot-swap)",
                    "model_id": model_id,
                    "model_name": new_config.name,
                }

            except Exception as e:
                print(f"[Pipeline] ✗ Hot-swap failed: {e}")
                traceback.print_exc()
                return {"success": False, "error": f"Hot-swap failed: {e}"}

        else:
            # ── Full restart: different landmark source or pipeline not running ──
            print(f"[Pipeline] Full restart required (landmark source: {old_landmark_source} → {new_landmark_source})")

            # Update registry
            self._registry.set_active_model(model_id)
            self._active_config = new_config
            self._settings.active_model = model_id

            if self._running:
                saved_settings = self._settings
                await self.stop()
                await self.start({
                    k: v for k, v in saved_settings.__dict__.items()
                })

            return {
                "success": True,
                "message": f"Switched to '{new_config.name}' (full restart)",
                "model_id": model_id,
                "model_name": new_config.name,
            }

    # ── Subsystem Initialization ────────────────

    async def _init_subsystems(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._init_subsystems_sync)

    def _init_subsystems_sync(self) -> None:
        settings = self._settings

        # ── Phase 6: Load active model config ──
        self._active_config = self._load_active_model_config()

        # ── 1. Camera ──
        try:
            from ..camera import CameraCapture
            self._camera = CameraCapture(
                camera_index=settings.camera_index,
                width=settings.resolution[0],
                height=settings.resolution[1],
                fps=settings.fps,
                mirror=False,
            )
            self._camera.start(
                on_disconnect=lambda: self._on_camera_disconnect()
            )
            print("[Pipeline] ✓ Camera started")
        except Exception as e:
            print(f"[Pipeline] ✗ Camera failed: {e}")
            self._camera = None

        # ── 2. Landmarker (config-driven) ──
        try:
            from ..recognition import Landmarker
            self._landmarker = Landmarker()

            if self._active_config:
                self._landmarker.init_from_config(self._active_config)
                print(f"[Pipeline] ✓ Landmarker ready (config: {self._active_config.input.landmark_source}, "
                      f"normalize: {self._active_config.input.normalize})")
            else:
                # Legacy: no config, use defaults
                self._landmarker._ensure_initialized()
                print("[Pipeline] ✓ Landmarker ready (default settings)")

            self._landmarker_ok = True
        except Exception as e:
            print(f"[Pipeline] ✗ Landmarker failed: {e}")
            self._landmarker = None
            self._landmarker_ok = False

        # ── 3. Recognizer (config-driven) ──
        try:
            from ..recognition import Recognizer

            # Determine model directory
            if self._active_config and self._registry:
                active_id = self._registry.active_id
                rec_model_dir = os.path.join(SIGN_MODEL_DIR, active_id)
            else:
                rec_model_dir = SIGN_MODEL_DIR

            rec_model_dir = os.path.normpath(rec_model_dir)

            if not os.path.isdir(rec_model_dir):
                os.makedirs(rec_model_dir, exist_ok=True)
                raise FileNotFoundError(f"No model files in {rec_model_dir}")

            self._recognizer = Recognizer(
                model_dir=rec_model_dir,
                min_confidence=settings.confidence_threshold,
                use_gpu=False,
            )

            # Phase 6: Pass config to recognizer
            if self._active_config:
                self._recognizer.load(config=self._active_config)
                print(f"[Pipeline] ✓ Recognizer loaded — model: '{self._active_config.name}'")
            else:
                self._recognizer.load()
                print("[Pipeline] ✓ Recognizer loaded (legacy mode)")

            self._model_loaded = True
            self._recognizer_ok = True

        except Exception as e:
            print(f"[Pipeline] ✗ Recognizer failed: {e}")
            print(f"[Pipeline]   Place your model file (.h5, .keras, .onnx) in:")
            print(f"[Pipeline]   {os.path.normpath(SIGN_MODEL_DIR)}")
            self._recognizer = None
            self._model_loaded = False
            self._recognizer_ok = False

        # ── 4. Compositor ──
        try:
            from ..camera import FrameCompositor
            self._compositor = FrameCompositor()
            print("[Pipeline] ✓ Compositor ready")
        except Exception as e:
            print(f"[Pipeline] ✗ Compositor failed: {e}")
            self._compositor = None

        # ── 5. Virtual Camera ──
        if settings.vcam_enabled:
            self._start_vcam()

        # ── 6. TTS ──
        if settings.tts_enabled:
            self._ensure_tts_ready()

        # ── 7. Virtual Mic (AFTER TTS so callback can be wired) ──
        if settings.vmic_enabled:
            self._start_vmic()

        # ── 8. Final wiring: ensure TTS→VMic callback is set ──
        self._wire_tts_to_vmic()

    def _load_active_model_config(self):
        """
        Load the active model's ModelConfig from the registry.
        Returns None if registry is not available or config fails to load.
        """
        if not self._registry:
            print("[Pipeline] ⚠ No model registry — using legacy model discovery")
            return None

        try:
            active_id = self._registry.active_id
            if not active_id:
                print("[Pipeline] ⚠ No active model set in registry")
                return None

            target = self._registry.get_model_by_id(active_id)
            if not target:
                print(f"[Pipeline] ⚠ Active model '{active_id}' not found in registry")
                return None

            from ..models import ModelConfig
            config = ModelConfig.load(target.config_path)
            print(f"[Pipeline] ✓ Model config loaded: '{config.name}' (id: {active_id})")
            return config

        except Exception as e:
            print(f"[Pipeline] ⚠ Failed to load model config: {e}")
            return None

    # ── TTS/VMic Wiring ─────────────────────────

    def _ensure_tts_ready(self) -> None:
        try:
            from ..speech import TTSEngine
            tts_dir = os.path.normpath(TTS_MODEL_DIR)

            if not os.path.isdir(tts_dir):
                os.makedirs(tts_dir, exist_ok=True)

            if not self._tts:
                self._tts = TTSEngine(model_dir=tts_dir)

            if not self._tts.is_loaded:
                self._tts.load(self._settings.tts_voice)

            if self._settings.audio_output_device:
                self._tts.set_local_device(self._settings.audio_output_device)
            else:
                self._tts.set_local_device("default")

            callback = self._make_vmic_callback()
            self._tts.start(callback=callback)

            print("[Pipeline] ✓ TTS ready")
        except Exception as e:
            print(f"[Pipeline] ✗ TTS failed: {e}")
            print(f"[Pipeline]   Place Piper voice files in: {os.path.normpath(TTS_MODEL_DIR)}")

    def _wire_tts_to_vmic(self) -> None:
        if self._tts and self._tts.is_loaded:
            callback = self._make_vmic_callback()
            self._tts.set_callback(callback)

    def _make_vmic_callback(self):
        def callback(audio, sr):
            if self._vmic and self._vmic.is_running:
                self._vmic.play(audio, sr)
                print(f"[Pipeline] TTS→VMic: sent {len(audio)} samples ({len(audio)/sr:.2f}s)")
            else:
                print(f"[Pipeline] TTS: synthesized {len(audio)} samples (VMic not active)")

        return callback

    # ── Camera Management ───────────────────────

    def _restart_camera(self) -> None:
        try:
            print("[Pipeline] Restarting camera with new settings...")
            if self._camera:
                self._camera.stop()
                self._camera = None

            from ..camera import CameraCapture
            self._camera = CameraCapture(
                camera_index=self._settings.camera_index,
                width=self._settings.resolution[0],
                height=self._settings.resolution[1],
                fps=self._settings.fps,
                mirror=False,
            )
            self._camera.start(
                on_disconnect=lambda: self._on_camera_disconnect()
            )
            self._camera_recovery_attempts = 0
            print(f"[Pipeline] ✓ Camera restarted — "
                  f"{self._settings.resolution[0]}x{self._settings.resolution[1]} "
                  f"@ {self._settings.fps}fps")

            if self._vcam_active:
                self._stop_vcam()
                self._start_vcam()

        except Exception as e:
            print(f"[Pipeline] ✗ Camera restart failed: {e}")

    def _on_camera_disconnect(self) -> None:
        print("[Pipeline] ⚠ Camera disconnected!")
        if self._running and self._camera_recovery_attempts < self._max_camera_recovery:
            import threading
            threading.Timer(3.0, self._try_recover_camera).start()

    def _try_recover_camera(self) -> None:
        if not self._running:
            return

        self._camera_recovery_attempts += 1
        attempt = self._camera_recovery_attempts
        print(f"[Pipeline] Camera recovery attempt {attempt}/{self._max_camera_recovery}...")

        try:
            if self._camera:
                self._camera.stop()
                self._camera = None

            import time as time_mod
            time_mod.sleep(1)

            from ..camera import CameraCapture
            self._camera = CameraCapture(
                camera_index=self._settings.camera_index,
                width=self._settings.resolution[0],
                height=self._settings.resolution[1],
                fps=self._settings.fps,
                mirror=False,
            )
            self._camera.start(
                on_disconnect=lambda: self._on_camera_disconnect()
            )
            self._camera_recovery_attempts = 0
            print(f"[Pipeline] ✓ Camera recovered on attempt {attempt}")

        except Exception as e:
            print(f"[Pipeline] ✗ Camera recovery failed ({attempt}/{self._max_camera_recovery}): {e}")
            if attempt >= self._max_camera_recovery:
                print("[Pipeline] ✗ Max camera recovery attempts reached")

    # ── Optional Subsystem Start/Stop ───────────

    def _start_vcam(self) -> None:
        try:
            from ..camera import VirtualCamera
            if self._vcam and self._vcam.is_running:
                return
            self._vcam = VirtualCamera(
                width=self._settings.resolution[0],
                height=self._settings.resolution[1],
                fps=self._settings.fps,
            )
            self._vcam.start()
            self._vcam_active = True
            print(f"[Pipeline] ✓ Virtual camera started (mirror: {'ON' if self._settings.vcam_mirror else 'OFF'})")
        except Exception as e:
            print(f"[Pipeline] ✗ Virtual camera failed: {e}")
            self._vcam = None
            self._vcam_active = False

    def _stop_vcam(self) -> None:
        if self._vcam:
            self._vcam.stop()
            self._vcam = None
        self._vcam_active = False

    def _start_vmic(self) -> None:
        try:
            from ..speech import VirtualMic

            if self._vmic and self._vmic.is_running:
                return

            self._vmic = VirtualMic()

            device = self._settings.vmic_device or None

            if device is None:
                cable_idx = VirtualMic.find_virtual_cable()
                if cable_idx is not None:
                    device = cable_idx
                    print(f"[Pipeline] VMic: auto-detected VB-Cable at device index {cable_idx}")
                else:
                    print("[Pipeline] ⚠ VMic: VB-Audio Virtual Cable not found!")
                    print("[Pipeline]   Install from: https://vb-audio.com/Cable/")
                    self._list_audio_devices()

            sample_rate = 22050
            if self._tts and self._tts.is_loaded:
                sample_rate = self._tts.sample_rate

            self._vmic.start(
                device=device,
                sample_rate=sample_rate,
                channels=1,
            )
            self._vmic_active = True
            print(f"[Pipeline] ✓ Virtual mic started → {self._vmic.device_name}")

            self._wire_tts_to_vmic()

        except Exception as e:
            print(f"[Pipeline] ✗ Virtual mic failed: {e}")
            self._list_audio_devices()
            self._vmic = None
            self._vmic_active = False

    def _list_audio_devices(self) -> None:
        try:
            import sounddevice as sd
            print("[Pipeline]   Available audio output devices:")
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_output_channels"] > 0:
                    marker = " ◄ CABLE" if "cable" in dev["name"].lower() else ""
                    print(f"[Pipeline]     [{i}] {dev['name']}{marker}")
        except Exception:
            print("[Pipeline]   (Could not enumerate audio devices)")

    def _stop_vmic(self) -> None:
        if self._vmic:
            self._vmic.stop()
            self._vmic = None
        self._vmic_active = False

    # ── Main Processing Loop ────────────────────

    async def _run_loop(self) -> None:
        frame_interval = 1.0 / max(self._settings.fps, 1)
        loop = asyncio.get_running_loop()

        try:
            while self._running:
                loop_start = time.time()

                try:
                    result = await loop.run_in_executor(
                        None, self._process_frame
                    )

                    if result is not None:
                        frame_b64, recognition_result = result

                        if frame_b64:
                            await self._broadcast_frame(frame_b64)

                        if recognition_result and recognition_result.hands_detected:
                            await self._broadcast_sign(recognition_result)

                        if recognition_result and recognition_result.transcript_changed:
                            await self._broadcast_transcript(recognition_result)

                        # ── TTS: speak on word completion ──
                        if recognition_result and self._settings.tts_enabled and self._tts:
                            self._handle_tts(recognition_result)

                        # SUCCESS
                        self._frames_processed += 1
                        self._fps_frame_count += 1
                        self._consecutive_errors = 0

                    # FPS calculation
                    now = time.time()
                    fps_elapsed = now - self._last_fps_time
                    if fps_elapsed >= 1.0:
                        self._fps = self._fps_frame_count / fps_elapsed
                        self._fps_frame_count = 0
                        self._last_fps_time = now
                        await self._broadcast_status()

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self._consecutive_errors += 1
                    err_msg = str(e)

                    if err_msg != self._last_error_msg:
                        print(
                            f"[Pipeline] Processing error "
                            f"({self._consecutive_errors}/{self.MAX_CONSECUTIVE_ERRORS}): {e}"
                        )
                        self._last_error_msg = err_msg

                    if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                        print("[Pipeline] Too many consecutive errors — auto-stopping")
                        await self._broadcast_error(
                            f"Pipeline auto-stopped: {err_msg}"
                        )
                        self._running = False
                        break

                # Frame rate limiting
                elapsed = time.time() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(0)

        except asyncio.CancelledError:
            print("[Pipeline] Processing loop cancelled")
            raise

    def _handle_tts(self, recognition_result) -> None:
        """
        Speak each NEW completed, spell-corrected word.
        Uses _corrected_words so TTS says the corrected version.
        """
        if not self._recognizer or not self._tts:
            return

        # Use CORRECTED words list, not raw words
        corrected_words = list(self._recognizer._corrected_words)
        word_count = len(corrected_words)

        # Speak any new words that were just completed (after spell correction)
        if word_count > self._last_spoken_word_count:
            new_words = corrected_words[self._last_spoken_word_count:]
            for word in new_words:
                word = word.strip()
                if not word:
                    continue
                print(f"[Pipeline] TTS speaking: '{word}'")
                self._tts.speak(word)
            self._last_spoken_word_count = word_count

    def _process_frame(self):
        """
        Single frame processing step (runs in thread pool).
        """
        import cv2

        if not self._camera:
            return None

        raw_frame = self._camera.read_and_clear()
        if raw_frame is None:
            return None

        # ── 1. Landmark extraction on raw frame ──
        hands_detected = False
        points = None
        handedness = None
        wrist_pos = None
        recognition_result = None

        if self._landmarker and self._landmarker_ok:
            success, raw_frame, points, wrist_pos, handedness = (
                self._landmarker.process(
                    raw_frame,
                    draw_landmarks=self._settings.show_landmarks,
                )
            )
            hands_detected = success
            self._hands_detected = hands_detected

        # ── 2. Recognition ──
        if self._recognizer and self._recognizer_ok and self._recognizer.is_loaded:
            recognition_result = self._recognizer.process(
                points=points,
                handedness=handedness,
                hands_detected=hands_detected,
            )

        # Overlay data
        transcript = ""
        sign = ""
        confidence = 0.0
        letter_added = False

        if recognition_result:
            transcript = recognition_result.full_transcript
            sign = recognition_result.letter
            confidence = recognition_result.confidence
            letter_added = recognition_result.letter_added

        # ── 3. Preview frame: mirror for selfie view, then draw overlays ──
        preview_frame = cv2.flip(raw_frame, 1)

        mirrored_wrist = None
        if wrist_pos is not None:
            mirrored_wrist = (1.0 - wrist_pos[0], wrist_pos[1])

        if self._compositor:
            preview_frame = self._compositor.render(
                frame=preview_frame,
                transcript=transcript,
                sign=sign,
                confidence=confidence,
                hands_detected=hands_detected,
                wrist_position=mirrored_wrist if hands_detected else None,
                letter_added=letter_added,
                show_overlay=self._settings.show_overlay,
                pipeline_running=True,
            )

        # ── 4. Virtual camera output ──
        if self._vcam and self._vcam.is_running:
            if self._settings.vcam_mirror:
                vcam_frame = preview_frame.copy()
            else:
                vcam_frame = raw_frame.copy()
                if self._compositor:
                    vcam_frame = self._compositor.render(
                        frame=vcam_frame,
                        transcript=transcript,
                        sign=sign,
                        confidence=confidence,
                        hands_detected=hands_detected,
                        wrist_position=wrist_pos if hands_detected else None,
                        letter_added=letter_added,
                        show_overlay=self._settings.show_overlay,
                        pipeline_running=True,
                    )
                self._vcam.send(vcam_frame)
            if self._settings.vcam_mirror:
                self._vcam.send(vcam_frame)

        # ── 5. Encode preview for WebSocket ──
        from ..camera import CameraCapture
        frame_b64 = CameraCapture.encode_base64(preview_frame, quality=70)

        return (frame_b64, recognition_result)

    # ── Broadcasting Helpers ────────────────────

    async def _broadcast_frame(self, frame_b64: str) -> None:
        if not self._broadcast_fn:
            return
        from .protocol import build_preview_frame
        await self._broadcast_fn(build_preview_frame(frame_b64))

    async def _broadcast_sign(self, result) -> None:
        if not self._broadcast_fn or not result.letter:
            return
        from .protocol import build_sign_detected
        msg = build_sign_detected(
            sign=result.letter,
            confidence=result.confidence,
            top_3=result.top_3,
            letter_added=result.letter_added,
            smoothed_confidence=result.smoothed_confidence,
        )
        await self._broadcast_fn(msg)

    async def _broadcast_transcript(self, result) -> None:
        if not self._broadcast_fn:
            return
        from .protocol import build_transcript_update
        msg = build_transcript_update(
            full_text=result.completed_text,
            latest_word=result.current_word,
            is_sentence_complete=result.is_sentence_complete,
        )
        await self._broadcast_fn(msg)

    async def _broadcast_status(self) -> None:
        if not self._broadcast_fn:
            return
        from .protocol import build_status_update
        msg = build_status_update(**self.get_status())
        await self._broadcast_fn(msg)

    async def _broadcast_error(self, message: str) -> None:
        if not self._broadcast_fn:
            return
        from .protocol import build_error
        await self._broadcast_fn(build_error(message))

    # ── Device Enumeration ──────────────────────

    @staticmethod
    def enumerate_cameras(max_check: int = 5) -> list[dict]:
        cameras: list[dict] = []
        try:
            import cv2
            for i in range(max_check):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    backend = cap.getBackendName()
                    cameras.append({
                        "index": i,
                        "name": f"Camera {i} ({backend})",
                    })
                    cap.release()
        except ImportError:
            pass
        except Exception as e:
            print(f"[Devices] Camera enumeration error: {e}")

        if not cameras:
            cameras = [{"index": 0, "name": "Default Camera"}]
        return cameras

    @staticmethod
    def enumerate_audio_devices() -> list[dict]:
        devices: list[dict] = []
        try:
            import sounddevice as sd
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_output_channels"] > 0:
                    devices.append({
                        "index": i,
                        "name": dev["name"],
                    })
        except ImportError:
            pass
        except Exception as e:
            print(f"[Devices] Audio enumeration error: {e}")
        return devices

    # ── Subsystem Release ───────────────────────

    async def _release_subsystems(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._release_subsystems_sync)

    def _release_subsystems_sync(self) -> None:
        if self._camera:
            try:
                self._camera.stop()
            except Exception as e:
                print(f"[Pipeline] Camera release error: {e}")
            self._camera = None

        if self._landmarker:
            try:
                self._landmarker.release()
            except Exception as e:
                print(f"[Pipeline] Landmarker release error: {e}")
            self._landmarker = None

        if self._recognizer:
            try:
                self._recognizer.release()
            except Exception as e:
                print(f"[Pipeline] Recognizer release error: {e}")
            self._recognizer = None
            self._model_loaded = False

        if self._compositor:
            self._compositor = None

        self._stop_vcam()

        if self._tts:
            try:
                self._tts.shutdown()
            except Exception as e:
                print(f"[Pipeline] TTS release error: {e}")
            self._tts = None

        self._stop_vmic()

        print("[Pipeline] All subsystems released")

    async def cleanup(self) -> None:
        if self._running:
            await self.stop()
        print("[Pipeline] All resources released")
