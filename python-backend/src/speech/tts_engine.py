"""
TTS Engine — Text-to-Speech with pyttsx3 (Windows) + Piper fallback
====================================================================
Converts text into audio numpy arrays.

Two audio outputs:
  1. VMic callback  → sends audio to virtual microphone (for Meet/Zoom)
  2. Local playback → plays audio through system speakers (for user to hear)

On Windows: Uses pyttsx3 (Windows SAPI voices) — works out of the box.
On Linux:   Uses Piper TTS (offline, natural voice) if piper_phonemize available.

IMPORTANT: pyttsx3 is NOT thread-safe across threads.
  - Engine must be created AND used on the SAME thread.
  - We create a fresh engine inside the synthesis thread.
"""

import os
import sys
import time
import wave
import tempfile
import threading
import queue
import traceback
from typing import Optional, Callable

import numpy as np


AudioCallback = Callable[[np.ndarray, int], None]


# ── Phonetic Expansion ─────────────────────────
# Single letters and short gibberish → pronounceable text

LETTER_TO_SPOKEN = {
    "A": "ay", "B": "bee", "C": "see", "D": "dee",
    "E": "ee", "F": "eff", "G": "jee", "H": "aitch",
    "I": "eye", "J": "jay", "K": "kay", "L": "ell",
    "M": "em", "N": "en", "O": "oh", "P": "pee",
    "Q": "cue", "R": "are", "S": "ess", "T": "tee",
    "U": "you", "V": "vee", "W": "double you", "X": "ex",
    "Y": "why", "Z": "zee",
}


def expand_for_speech(text: str) -> str:
    """Expand single letters or short non-words into pronounceable text."""
    text = text.strip()
    if not text:
        return text

    upper = text.upper()
    vowels = set("AEIOU")
    has_vowel = any(c in vowels for c in upper if c.isalpha())

    # If it looks like a real word (3+ chars with vowels), keep as-is
    if len(text) >= 3 and has_vowel:
        return text.lower()

    # Spell it out letter by letter
    parts = []
    for ch in upper:
        spoken = LETTER_TO_SPOKEN.get(ch)
        if spoken:
            parts.append(spoken)
        elif ch == " ":
            continue
        else:
            parts.append(ch.lower())

    if parts:
        return ", ".join(parts)

    return text.lower()


class TTSEngine:

    DEFAULT_VOICE = "en_US-lessac-medium"
    SAMPLE_RATE = 22050

    def __init__(self, model_dir: str = ""):
        self._model_dir = model_dir
        self._sample_rate = self.SAMPLE_RATE
        self._loaded = False

        # Backend state
        self._backend = ""  # "pyttsx3" or "piper"
        self._voice = None  # Piper voice object (if using piper)
        self._voice_name = ""

        # Threading
        self._queue: queue.Queue = queue.Queue(maxsize=20)
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # VMic callback
        self._callback: Optional[AudioCallback] = None
        self._callback_lock = threading.Lock()

        # Local playback
        self._local_device_name: str = ""
        self._local_device_index: Optional[int] = None
        self._local_enabled: bool = False
        self._local_lock = threading.Lock()

    # ── Properties ──────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def voice_name(self) -> str:
        return self._voice_name

    @property
    def has_callback(self) -> bool:
        with self._callback_lock:
            return self._callback is not None

    # ── Loading ─────────────────────────────────

    def load(self, voice_name: Optional[str] = None) -> None:
        """
        Load TTS backend.
        On Windows: tries pyttsx3 first (always works).
        On Linux:   tries Piper first (needs piper_phonemize).
        """
        voice_name = voice_name or self.DEFAULT_VOICE

        if sys.platform == "win32":
            if self._try_pyttsx3():
                return
            if self._try_piper(voice_name):
                return
        else:
            if self._try_piper(voice_name):
                return
            if self._try_pyttsx3():
                return

        self._loaded = False
        print("[TTS] ✗ No TTS backend available!")
        print("[TTS]   Windows: pip install pyttsx3")
        print("[TTS]   Linux:   pip install piper-tts piper-phonemize")

    def _try_pyttsx3(self) -> bool:
        """Test pyttsx3 on the CURRENT thread."""
        print("[TTS] Testing pyttsx3 (system voices)...")
        try:
            import pyttsx3

            engine = pyttsx3.init()
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            try:
                engine.save_to_file("test", temp_path)
                engine.runAndWait()

                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 100:
                    with wave.open(temp_path, "rb") as wr:
                        sr = wr.getframerate()
                        n = wr.getnframes()
                        if sr > 0:
                            self._sample_rate = sr
                        print(f"[TTS]   pyttsx3 test: {n} samples at {sr}Hz")

                    if n > 0:
                        self._backend = "pyttsx3"
                        self._loaded = True
                        self._voice_name = "system"
                        print(f"[TTS] ✓ Backend: pyttsx3 (Windows SAPI) — rate: {self._sample_rate}Hz")
                        try:
                            engine.stop()
                        except Exception:
                            pass
                        return True
                    else:
                        print("[TTS]   pyttsx3 produced 0 frames")
                else:
                    size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                    print(f"[TTS]   pyttsx3 WAV too small: {size} bytes")

            finally:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                try:
                    engine.stop()
                except Exception:
                    pass

        except ImportError:
            print("[TTS]   pyttsx3 not installed: pip install pyttsx3")
        except Exception as e:
            print(f"[TTS]   pyttsx3 test failed: {e}")
            traceback.print_exc()

        return False

    def _try_piper(self, voice_name: str) -> bool:
        """Try to load Piper TTS and test synthesis."""
        try:
            from piper import PiperVoice
        except ImportError:
            print("[TTS]   Piper not installed")
            return False

        model_path = self._find_voice_model(voice_name)
        if not model_path:
            print(f"[TTS]   Piper voice not found: {voice_name}")
            return False

        print(f"[TTS] Loading Piper voice: {model_path}")
        start = time.time()

        try:
            voice = PiperVoice.load(model_path)
            elapsed = time.time() - start

            if hasattr(voice, "config") and hasattr(voice.config, "sample_rate"):
                self._sample_rate = voice.config.sample_rate

            print(f"[TTS]   Piper loaded in {elapsed:.2f}s — rate: {self._sample_rate}Hz")

            audio = self._piper_synthesize_test(voice, "hello")
            if audio is not None and len(audio) > 100:
                self._voice = voice
                self._voice_name = voice_name
                self._backend = "piper"
                self._loaded = True
                print(f"[TTS] ✓ Backend: Piper — {len(audio)} test samples")
                return True
            else:
                n = len(audio) if audio is not None else 0
                print(f"[TTS]   Piper synthesis test: {n} samples (too few)")
                return False

        except Exception as e:
            print(f"[TTS]   Piper load failed: {e}")
            return False

    def _piper_synthesize_test(self, voice, text: str) -> Optional[np.ndarray]:
        """Test Piper synthesis with temp file method."""
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                voice.synthesize(text, wf)

            with wave.open(temp_path, "rb") as wr:
                sr = wr.getframerate()
                if sr > 0:
                    self._sample_rate = sr
                n = wr.getnframes()
                if n == 0:
                    return None
                raw = wr.readframes(n)
                return np.frombuffer(raw, dtype=np.int16).copy()
        except Exception as e:
            print(f"[TTS]   Piper test error: {e}")
            return None
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _find_voice_model(self, voice_name: str) -> Optional[str]:
        """Find Piper .onnx voice model file."""
        if not self._model_dir:
            return None

        candidates = [
            os.path.join(self._model_dir, f"{voice_name}.onnx"),
            os.path.join(self._model_dir, voice_name),
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        if os.path.isdir(self._model_dir):
            for fn in os.listdir(self._model_dir):
                if voice_name in fn and fn.endswith(".onnx") and not fn.endswith(".onnx.json"):
                    return os.path.join(self._model_dir, fn)

        return None

    # ── VMic Callback ───────────────────────────

    def set_callback(self, callback: AudioCallback) -> None:
        with self._callback_lock:
            self._callback = callback
        print("[TTS] ✓ VMic callback updated")

    def _get_callback(self) -> Optional[AudioCallback]:
        with self._callback_lock:
            return self._callback

    # ── Local Speaker Playback ──────────────────

    def set_local_device(self, device_name: str) -> None:
        """
        Set the local audio output device for TTS playback.
        
        Args:
            device_name: Device name, "default" for system default, or "" to disable.
        """
        with self._local_lock:
            self._local_device_name = device_name
            self._local_device_index = None

            if device_name == "default":
                # Use system default output device
                self._local_enabled = True
                self._local_device_index = None  # sounddevice uses None = default
                print("[TTS] ✓ Local playback → system default speaker")
            elif device_name:
                self._local_device_index = self._find_device_index(device_name)
                self._local_enabled = True
                if self._local_device_index is not None:
                    print(f"[TTS] ✓ Local playback → '{device_name}' (idx: {self._local_device_index})")
                else:
                    print(f"[TTS] ⚠ Local device '{device_name}' not found — using default")
            else:
                self._local_enabled = False
                print("[TTS] Local playback disabled")

    def _find_device_index(self, name: str) -> Optional[int]:
        """Find sounddevice output device index by name."""
        try:
            import sounddevice as sd
            name_lower = name.lower()
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_output_channels"] > 0:
                    if name_lower in dev["name"].lower() or dev["name"].lower() in name_lower:
                        return i
        except Exception as e:
            print(f"[TTS]   Device lookup error: {e}")
        return None

    def _play_local(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play audio through local speakers using sounddevice."""
        with self._local_lock:
            if not self._local_enabled:
                return
            device_idx = self._local_device_index
            device_name = self._local_device_name

        try:
            import sounddevice as sd

            # Convert to float32 for sounddevice
            audio_float = audio.astype(np.float32) / 32768.0

            # Play synchronously (we're on synthesis thread, it's fine)
            sd.play(
                audio_float,
                samplerate=sample_rate,
                device=device_idx,  # None = default device
                blocking=True,
            )

            print(f"[TTS] ✓ Local playback: {len(audio)} samples → '{device_name or 'default'}'")

        except Exception as e:
            print(f"[TTS] ✗ Local playback error: {e}")

    # ── Synthesis Thread ────────────────────────

    def start(self, callback: Optional[AudioCallback] = None) -> None:
        if callback:
            self.set_callback(callback)

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._synthesis_loop, daemon=True)
        self._thread.start()
        print("[TTS] Synthesis thread started")

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        try:
            self._queue.put_nowait(None)  # Wake up the thread
        except queue.Full:
            pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        print("[TTS] Synthesis thread stopped")

    def speak(self, text: str) -> None:
        """Queue text for background synthesis. Non-blocking."""
        if not self._running:
            print("[TTS] ⚠ Cannot speak — thread not running")
            return

        if not text or not text.strip():
            return

        try:
            self._queue.put_nowait(text)
            print(f"[TTS] Queued: '{text}'")
        except queue.Full:
            print("[TTS] Queue full — dropping")

    def _synthesis_loop(self) -> None:
        """
        Background synthesis thread.

        CRITICAL: pyttsx3 engine must be created HERE (same thread as runAndWait).
        """
        # Create pyttsx3 engine on THIS thread
        thread_engine = None
        if self._backend == "pyttsx3":
            try:
                import pyttsx3
                thread_engine = pyttsx3.init()
                print("[TTS] ✓ pyttsx3 engine created on synthesis thread")
            except Exception as e:
                print(f"[TTS] ✗ pyttsx3 init on thread failed: {e}")
                traceback.print_exc()
                return

        total_spoken = 0

        while self._running:
            try:
                text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if text is None:
                continue

            text = text.strip()
            if not text:
                continue

            # Expand for speech
            spoken_text = expand_for_speech(text)
            if spoken_text != text.lower():
                print(f"[TTS] Expanded '{text}' → '{spoken_text}'")

            # Synthesize
            audio = None
            try:
                if self._backend == "pyttsx3" and thread_engine:
                    audio = self._do_pyttsx3_synth(thread_engine, spoken_text)
                elif self._backend == "piper" and self._voice:
                    audio = self._do_piper_synth(spoken_text)
            except Exception as e:
                print(f"[TTS] ✗ Synthesis error for '{text}': {e}")
                traceback.print_exc()
                continue

            if audio is not None and len(audio) > 0:
                total_spoken += 1
                print(
                    f"[TTS] ✓ Synthesized '{text}' — "
                    f"{len(audio)} samples ({len(audio)/self._sample_rate:.2f}s)"
                )

                # ── Output 1: VMic callback (for Google Meet / Zoom) ──
                callback = self._get_callback()
                if callback:
                    try:
                        callback(audio, self._sample_rate)
                        print("[TTS] ✓ Sent to VMic")
                    except Exception as e:
                        print(f"[TTS] ✗ VMic callback error: {e}")

                # ── Output 2: Local speaker playback (for user to hear) ──
                self._play_local(audio, self._sample_rate)

            else:
                print(f"[TTS] ✗ No audio produced for '{text}'")

        # Cleanup thread-local engine
        if thread_engine:
            try:
                thread_engine.stop()
            except Exception:
                pass

        print(f"[TTS] Synthesis thread exiting — spoke {total_spoken} utterances")

    # ── pyttsx3 Synthesis (called on synthesis thread) ──

    def _do_pyttsx3_synth(self, engine, text: str) -> Optional[np.ndarray]:
        """Synthesize using pyttsx3 — engine MUST be on this thread."""
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            engine.save_to_file(text, temp_path)
            engine.runAndWait()

            if not os.path.exists(temp_path):
                print("[TTS]   pyttsx3: temp file not created")
                return None

            size = os.path.getsize(temp_path)
            if size < 100:
                print(f"[TTS]   pyttsx3: WAV too small ({size} bytes)")
                return None

            with wave.open(temp_path, "rb") as wr:
                sr = wr.getframerate()
                nch = wr.getnchannels()
                sw = wr.getsampwidth()
                n_frames = wr.getnframes()

                if n_frames == 0:
                    print("[TTS]   pyttsx3: 0 frames in WAV")
                    return None

                raw = wr.readframes(n_frames)

                # Convert to int16
                if sw == 2:
                    audio = np.frombuffer(raw, dtype=np.int16)
                elif sw == 1:
                    audio = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) * 256 - 32768
                elif sw == 4:
                    audio = (np.frombuffer(raw, dtype=np.int32) >> 16).astype(np.int16)
                else:
                    print(f"[TTS]   pyttsx3: unsupported sample width {sw}")
                    return None

                # Convert stereo to mono
                if nch == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
                elif nch > 2:
                    audio = audio.reshape(-1, nch)[:, 0].astype(np.int16)

                # Resample if needed
                if sr > 0 and sr != self._sample_rate:
                    audio = self._resample(audio, sr, self._sample_rate)

                return audio.copy()

        except Exception as e:
            print(f"[TTS]   pyttsx3 synth error: {e}")
            traceback.print_exc()
            return None
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    # ── Piper Synthesis (called on synthesis thread) ──

    def _do_piper_synth(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using Piper TTS (temp file method)."""
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                self._voice.synthesize(text, wf)

            with wave.open(temp_path, "rb") as wr:
                sr = wr.getframerate()
                if sr > 0:
                    self._sample_rate = sr
                n = wr.getnframes()
                if n == 0:
                    return None
                raw = wr.readframes(n)
                return np.frombuffer(raw, dtype=np.int16).copy()

        except Exception as e:
            print(f"[TTS]   Piper synth error: {e}")
            return None
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    # ── Audio Utils ─────────────────────────────

    @staticmethod
    def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple linear resampling."""
        if from_rate == to_rate:
            return audio

        ratio = to_rate / from_rate
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio.astype(np.float64)).astype(np.int16)

    # ── Cleanup ─────────────────────────────────

    def shutdown(self) -> None:
        self.stop()
        self._voice = None
        self._loaded = False
        self._backend = ""
        with self._callback_lock:
            self._callback = None
        with self._local_lock:
            self._local_enabled = False
            self._local_device_index = None
        print("[TTS] Engine shut down")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
