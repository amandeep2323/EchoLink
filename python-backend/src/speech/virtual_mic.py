"""
Virtual Microphone — Audio Output to VB-Audio Virtual Cable
==============================================================
Plays synthesized TTS audio through a virtual audio device
(VB-Audio Virtual Cable) so that other applications like
Google Meet, Zoom, and Teams receive the TTS audio as
microphone input.

How it works:
  1. TTS engine synthesizes text → numpy audio array
  2. VirtualMic sends audio to "CABLE Input (VB-Audio Virtual Cable)"
  3. In Google Meet/Zoom, select "CABLE Output" as your microphone
  4. Other participants hear the TTS audio

Prerequisites:
  - VB-Audio Virtual Cable installed: https://vb-audio.com/Cable/
  - sounddevice installed: pip install sounddevice

Setup Guide:
  1. Install VB-Audio Virtual Cable from https://vb-audio.com/Cable/
  2. Restart your computer after install
  3. In SignSpeak: enable VMic — it auto-detects "CABLE Input"
  4. In Google Meet: Settings → Audio → Microphone → "CABLE Output (VB-Audio Virtual Cable)"
  5. Enable TTS in SignSpeak → completed sentences are spoken through the virtual mic

Usage:
    vmic = VirtualMic()
    vmic.start(device="CABLE Input")  # or device index, or None for auto-detect
    
    vmic.play(audio_array, sample_rate=22050)
    
    vmic.stop()
"""

import threading
import queue
import time
from typing import Optional, Union

import numpy as np


class VirtualMic:
    """
    Outputs audio through a virtual microphone device via sounddevice.
    Typically targets VB-Audio Virtual Cable for use in video conferencing.
    """

    # Audio queue settings
    MAX_QUEUE_SIZE = 10
    CHUNK_SIZE = 4096  # Samples per write

    # Search patterns for VB-Audio Virtual Cable
    CABLE_PATTERNS = [
        "cable input",
        "virtual cable",
        "vb-audio",
        "vb-cable",
    ]

    def __init__(self):
        # State
        self._stream = None        # sounddevice.OutputStream
        self._running = False
        self._device_index: Optional[int] = None
        self._device_name: str = ""
        self._sample_rate: int = 22050
        self._channels: int = 1

        # Audio queue for non-blocking playback
        self._audio_queue: queue.Queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Stats
        self._samples_played: int = 0
        self._chunks_played: int = 0

    # ── Properties ──────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def samples_played(self) -> int:
        return self._samples_played

    # ── Lifecycle ───────────────────────────────

    def start(
        self,
        device: Optional[Union[int, str]] = None,
        sample_rate: int = 22050,
        channels: int = 1,
    ) -> None:
        """
        Start the virtual microphone output.
        
        Args:
            device: Audio output device — name (str) or index (int).
                    If None, auto-detects VB-Audio Virtual Cable.
            sample_rate: Audio sample rate (should match TTS output).
            channels: Number of audio channels (1=mono, 2=stereo).
        """
        if self._running:
            return

        try:
            import sounddevice as sd
        except ImportError as e:
            raise RuntimeError(
                f"sounddevice not installed: {e}\n"
                f"Install with: pip install sounddevice"
            ) from e

        self._sample_rate = sample_rate
        self._channels = channels

        # Resolve device (auto-detect VB-Cable if not specified)
        if device is None:
            self._device_index = self.find_virtual_cable()
            if self._device_index is None:
                print("[VMic] ⚠ VB-Audio Virtual Cable not detected — using default output")
                print("[VMic]   To fix: Install VB-Audio from https://vb-audio.com/Cable/")
                print("[VMic]   After install, restart your computer")
        elif isinstance(device, int):
            self._device_index = device
        elif isinstance(device, str) and device.strip():
            self._device_index = self._find_device_by_name(device)
        else:
            self._device_index = None

        # Get device info
        if self._device_index is not None:
            try:
                dev_info = sd.query_devices(self._device_index)
                self._device_name = dev_info["name"]
            except Exception:
                self._device_name = f"Device {self._device_index}"
        else:
            self._device_name = "Default Output"

        print(
            f"[VMic] Starting — device: '{self._device_name}' "
            f"(idx: {self._device_index}), "
            f"rate: {self._sample_rate}Hz, ch: {self._channels}"
        )

        # Start the playback thread
        self._running = True
        self._samples_played = 0
        self._chunks_played = 0
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the virtual microphone output."""
        if not self._running:
            return

        self._running = False

        # Unblock the queue
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            pass

        # Wait for thread
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        # Close stream
        with self._lock:
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        duration = self._samples_played / self._sample_rate if self._sample_rate > 0 else 0
        print(
            f"[VMic] Stopped — {self._samples_played:,} samples played "
            f"({duration:.1f}s audio, {self._chunks_played} chunks)"
        )

    # ── Audio Playback ──────────────────────────

    def play(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """
        Queue audio for playback through the virtual mic.
        Non-blocking — audio will play in the background thread.
        
        Args:
            audio: Numpy array of audio samples (int16 or float32)
            sample_rate: Sample rate of the audio. If different from
                         the stream rate, audio will be resampled.
        """
        if not self._running:
            return

        # Convert to float32 if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Simple resampling if rates don't match
        if sample_rate and sample_rate != self._sample_rate:
            audio = self._resample(audio, sample_rate, self._sample_rate)

        # Ensure correct shape (samples, channels)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        try:
            self._audio_queue.put_nowait(audio)
        except queue.Full:
            print("[VMic] Queue full — dropping audio")

    # ── Playback Thread ─────────────────────────

    def _playback_loop(self) -> None:
        """Background thread that plays queued audio through sounddevice."""
        try:
            import sounddevice as sd
        except ImportError:
            print("[VMic] sounddevice not available — thread exiting")
            return

        try:
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                device=self._device_index,
                dtype="float32",
                blocksize=self.CHUNK_SIZE,
            )
            self._stream.start()
            print(f"[VMic] ✓ Audio stream opened on '{self._device_name}'")
        except Exception as e:
            print(f"[VMic] ✗ Failed to open output stream: {e}")
            if self._device_index is not None:
                print(f"[VMic]   Device index {self._device_index} may not support "
                      f"{self._sample_rate}Hz / {self._channels}ch")
                print(f"[VMic]   Try selecting a different device in Settings")
            self._running = False
            return

        while self._running:
            try:
                # Get next audio from queue
                audio = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if audio is None:
                continue  # Sentinel

            # Play audio in chunks
            try:
                pos = 0
                total = len(audio)
                while pos < total and self._running:
                    chunk_end = min(pos + self.CHUNK_SIZE, total)
                    chunk = audio[pos:chunk_end]

                    self._stream.write(chunk)
                    self._samples_played += len(chunk)
                    self._chunks_played += 1
                    pos = chunk_end

            except Exception as e:
                print(f"[VMic] Playback error: {e}")

        # Clean up
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None

    # ── Device Discovery ────────────────────────

    @classmethod
    def find_virtual_cable(cls) -> Optional[int]:
        """
        Find VB-Audio Virtual Cable in the system's audio devices.
        
        Searches for devices matching common VB-Cable names.
        Returns the device index or None if not found.
        """
        try:
            import sounddevice as sd
            devices = sd.query_devices()

            # First pass: look for "CABLE Input" specifically (most reliable)
            for i, dev in enumerate(devices):
                name = dev["name"].lower()
                if "cable input" in name and dev["max_output_channels"] > 0:
                    print(f"[VMic] ✓ Found VB-Cable: '{dev['name']}' (idx: {i})")
                    return i

            # Second pass: broader search
            for i, dev in enumerate(devices):
                name = dev["name"].lower()
                if dev["max_output_channels"] > 0:
                    for pattern in cls.CABLE_PATTERNS:
                        if pattern in name:
                            print(f"[VMic] ✓ Found Virtual Cable: '{dev['name']}' (idx: {i})")
                            return i

        except Exception as e:
            print(f"[VMic] Device enumeration error: {e}")

        return None

    @staticmethod
    def _find_device_by_name(name: str) -> Optional[int]:
        """Find a device by name (partial, case-insensitive match)."""
        try:
            import sounddevice as sd
            for i, dev in enumerate(sd.query_devices()):
                if name.lower() in dev["name"].lower():
                    if dev["max_output_channels"] > 0:
                        return i
        except Exception:
            pass

        print(f"[VMic] Device '{name}' not found — using default")
        return None

    @staticmethod
    def list_output_devices() -> list[dict]:
        """List all available audio output devices."""
        devices = []
        try:
            import sounddevice as sd
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_output_channels"] > 0:
                    is_cable = "cable" in dev["name"].lower()
                    devices.append({
                        "index": i,
                        "name": dev["name"],
                        "channels": dev["max_output_channels"],
                        "default_rate": dev["default_samplerate"],
                        "is_virtual_cable": is_cable,
                    })
        except Exception:
            pass
        return devices

    # ── Resampling ──────────────────────────────

    @staticmethod
    def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple linear resampling (for basic rate conversion)."""
        if from_rate == to_rate:
            return audio

        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)

        # Linear interpolation
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, new_length)

        if audio.ndim == 1:
            return np.interp(x_new, x_old, audio).astype(audio.dtype)
        else:
            result = np.zeros((new_length, audio.shape[1]), dtype=audio.dtype)
            for ch in range(audio.shape[1]):
                result[:, ch] = np.interp(x_new, x_old, audio[:, ch])
            return result

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
