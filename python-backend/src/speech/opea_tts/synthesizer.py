# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from opea-project/GenAIComps for desktop integration
# Original: https://github.com/opea-project/GenAIComps/tree/main/comps/tts
#
# TTSEngine-compatible synthesizer wrapper for OPEA TTS

"""
OPEA TTS Synthesizer

TTSEngine-compatible interface wrapper for Intel OPEA SpeechT5 synthesis.
Provides automatic model downloading, backend detection, and audio synthesis
conforming to the existing TTSEngine API.

This is the main entry point for using OPEA TTS in the EchoLink application.
"""

import os
import time
from typing import Optional

import numpy as np

from .backend_detector import detect_inference_backend
from .speecht5_core import SpeechT5ModelCore
from .model_downloader import download_models_if_needed


class OpeaTtsSynthesizer:
    """
    TTSEngine-compatible interface for Intel OPEA SpeechT5.
    
    This class wraps the OPEA SpeechT5ModelCore to provide a simple API
    compatible with EchoLink's TTSEngine. It handles model downloading,
    backend detection, and audio synthesis.
    
    Attributes:
        model_dir: Directory for model cache storage
        sample_rate: Audio sample rate (always 16000 Hz for SpeechT5)
        core: SpeechT5ModelCore instance (initialized after load())
    
    Example:
        >>> synth = OpeaTtsSynthesizer(model_dir="models/tts/opea_speecht5")
        >>> if synth.load():
        ...     audio = synth.synthesize("Hello world")
        ...     print(f"Generated {len(audio)} audio samples")
    """
    
    # SpeechT5 native sample rate
    SAMPLE_RATE = 16000
    
    def __init__(self, model_dir: str = ""):
        """
        Initialize OPEA TTS synthesizer.
        
        Args:
            model_dir: Directory for model cache. 
                       Defaults to "models/tts/opea_speecht5" if empty.
        """
        self.model_dir = model_dir or "models/tts/opea_speecht5"
        self.sample_rate = self.SAMPLE_RATE
        self.core: Optional[SpeechT5ModelCore] = None
        self._loaded = False
        self._backend = "unknown"
    
    @property
    def is_loaded(self) -> bool:
        """Check if synthesizer is initialized and ready."""
        return self._loaded
    
    @property
    def backend_name(self) -> str:
        """Get the active inference backend name."""
        if self._backend == "openvino":
            return "OpenVINO"
        elif self._backend == "cpu":
            return "PyTorch CPU"
        else:
            return self._backend
    
    def load(self, run_test: bool = False) -> bool:
        """
        Load models and initialize synthesis pipeline.
        
        Automatically downloads required models from HuggingFace on first run.
        Subsequent runs reuse cached models.
        
        Args:
            run_test: If True, run a one-off test synthesis to validate the
                      pipeline. Off by default to keep load time low — the
                      first real synthesis will surface any issue anyway.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print("[OPEA TTS] Loading Intel OPEA SpeechT5...")
            start_time = time.time()
            
            # Step 1: Download models if needed (automatic setup)
            success, message = download_models_if_needed(self.model_dir)
            if not success:
                print(f"[OPEA TTS] ✗ Model setup failed: {message}")
                return False
            
            # Step 2: Detect inference backend
            self._backend = detect_inference_backend()
            print(f"[OPEA TTS] Backend: {self.backend_name}")
            
            # Step 3: Initialize core model
            self.core = SpeechT5ModelCore(
                model_dir=self.model_dir,
                backend=self._backend
            )
            
            if not self.core.initialize():
                print("[OPEA TTS] ✗ Core initialization failed")
                return False
            
            # Mark as loaded so synthesize()'s readiness guard passes.
            self._loaded = True
            
            # Step 4 (optional): Test synthesis. Skipped by default for speed.
            if run_test:
                print("[OPEA TTS] Running test synthesis...")
                test_audio = self.synthesize("test", voice="default")
                if test_audio is None or len(test_audio) < 100:
                    print(f"[OPEA TTS] ✗ Test synthesis failed (audio length: {len(test_audio) if test_audio is not None else 0})")
                    self._loaded = False
                    return False
            
            elapsed = time.time() - start_time
            print(f"[OPEA TTS] ✓ Loaded successfully in {elapsed:.2f}s")
            print(f"[OPEA TTS]   Backend: {self.backend_name}")
            print(f"[OPEA TTS]   Sample rate: {self.sample_rate}Hz")
            
            return True
            
        except Exception as e:
            print(f"[OPEA TTS] ✗ Load failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def synthesize(self, text: str, voice: str = "default") -> Optional[np.ndarray]:
        """
        Synthesize speech from text.
        
        Converts input text to audio waveform using SpeechT5 model.
        Supports long text via automatic batch processing.
        
        Args:
            text: Input text to synthesize (any length, auto-batched)
            voice: Speaker voice - "default" or "male"
        
        Returns:
            Audio as numpy.ndarray with:
            - dtype: int16
            - shape: (samples,)
            - sample rate: 16000 Hz
            Returns None on synthesis failure
            
        Raises:
            RuntimeError: If synthesizer not loaded (call load() first)
            ValueError: If voice parameter is invalid
            
        Example:
            >>> synth = OpeaTtsSynthesizer()
            >>> synth.load()
            >>> audio = synth.synthesize("Hello world", voice="male")
            >>> print(audio.shape, audio.dtype)
            (24000,) int16
        """
        # Validation
        if not self._loaded or self.core is None:
            raise RuntimeError("OPEA TTS not loaded — call load() first")
        
        # Handle empty input
        if not text or not text.strip():
            return np.array([], dtype=np.int16)
        
        # Validate voice
        if voice not in ["default", "male"]:
            raise ValueError(f"Invalid voice: {voice}. Use 'default' or 'male'")
        
        # Truncate long text with warning
        MAX_LENGTH = 1000
        if len(text) > MAX_LENGTH:
            print(f"[OPEA TTS] Warning: Text exceeds {MAX_LENGTH} characters, truncating")
            text = text[:MAX_LENGTH]
        
        # Validate UTF-8 encoding
        try:
            text.encode('utf-8')
        except UnicodeEncodeError as e:
            print(f"[OPEA TTS] ✗ Text encoding error: {e}")
            return None
        
        # Synthesize
        try:
            synthesis_start = time.time()
            
            # Core synthesis returns float32 audio
            audio_float = self.core.text_to_speech(text, voice=voice)
            
            if audio_float is None or len(audio_float) == 0:
                print("[OPEA TTS] ✗ Core synthesis returned no audio")
                return None
            
            # Convert float32 [-1, 1] to int16 [-32768, 32767] for TTSEngine compatibility
            audio_int16 = (audio_float * 32767).astype(np.int16)
            
            synthesis_time = time.time() - synthesis_start
            duration_sec = len(audio_int16) / self.sample_rate
            
            print(f"[OPEA TTS] ✓ Synthesized {len(text)} chars → {len(audio_int16)} samples ({duration_sec:.2f}s audio) in {synthesis_time:.2f}s")
            
            # Performance warning
            if synthesis_time > 5.0:
                print(f"[OPEA TTS] ⚠ Synthesis took {synthesis_time:.2f}s (consider using OpenVINO for acceleration)")
            
            return audio_int16
            
        except Exception as e:
            print(f"[OPEA TTS] ✗ Synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def shutdown(self) -> None:
        """
        Clean up resources.
        
        Releases model references and clears state. After calling shutdown(),
        the synthesizer must be reloaded with load() before use.
        """
        self.core = None
        self._loaded = False
        self._backend = "unknown"
        print("[OPEA TTS] Shutdown complete")
