# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from opea-project/GenAIComps for desktop integration
# Original: https://github.com/opea-project/GenAIComps/tree/main/comps/tts
#
# Model downloader with automatic setup and caching

"""
Model Downloader Module

Handles automatic downloading and caching of HuggingFace models required
for OPEA TTS synthesis. Downloads models on first use and reuses cached
versions on subsequent launches.

Models downloaded:
- microsoft/speecht5_tts (main TTS model)
- microsoft/speecht5_hifigan (vocoder)
- Speaker embeddings from Intel Extension for Transformers
"""

import os
from typing import Tuple

def download_models_if_needed(model_dir: str) -> Tuple[bool, str]:
    """
    Download SpeechT5 models if not cached, otherwise use existing cache.
    
    This function checks for the presence of required model files in the cache
    directory. If models are missing, it triggers downloads from HuggingFace
    using the transformers library's automatic caching mechanism.
    
    Args:
        model_dir: Path to model cache directory
        
    Returns:
        Tuple of (success: bool, message: str)
        - success: True if models are available (cached or downloaded successfully)
        - message: Status message describing what happened
        
    Side Effects:
        - Creates model_dir if it doesn't exist
        - Downloads models from HuggingFace (first run only, ~200MB)
        - Prints progress messages to console
        
    Example:
        >>> success, msg = download_models_if_needed("models/tts/opea_speecht5")
        [OPEA TTS] Checking model cache...
        [OPEA TTS] Models already cached — ready to use
        >>> print(success, msg)
        True "Models cached at models/tts/opea_speecht5"
    """
    try:
        # Create cache directory
        os.makedirs(model_dir, exist_ok=True)
        
        print("[OPEA TTS] Checking model cache...")
        
        # Check for cached models using HuggingFace cache structure
        cache_markers = [
            "models--microsoft--speecht5_tts",
            "models--microsoft--speecht5_hifigan"
        ]
        
        models_cached = all(
            os.path.exists(os.path.join(model_dir, marker))
            for marker in cache_markers
        )
        
        if models_cached:
            print("[OPEA TTS] ✓ Models already cached — ready to use")
            return True, f"Models cached at {model_dir}"
        
        # Models not cached — trigger download via transformers
        print("[OPEA TTS] Models not found in cache — downloading...")
        print("[OPEA TTS] This may take a few minutes on first run (~200MB download)")
        
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        
        # Download processor (triggers model download with progress bar)
        print("[OPEA TTS] Downloading SpeechT5 processor...")
        SpeechT5Processor.from_pretrained(
            "microsoft/speecht5_tts",
            cache_dir=model_dir
        )
        
        # Download main TTS model
        print("[OPEA TTS] Downloading SpeechT5 TTS model...")
        SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts",
            cache_dir=model_dir
        )
        
        # Download vocoder
        print("[OPEA TTS] Downloading SpeechT5 HiFi-GAN vocoder...")
        SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan",
            cache_dir=model_dir
        )
        
        print("[OPEA TTS] ✓ Model download complete")
        return True, f"Models downloaded to {model_dir}"
        
    except ImportError as e:
        error_msg = f"transformers library not installed: {e}"
        print(f"[OPEA TTS] ✗ {error_msg}")
        print("[OPEA TTS]   Install with: pip install transformers torch")
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Model download failed: {e}"
        print(f"[OPEA TTS] ✗ {error_msg}")
        print("[OPEA TTS]   Check internet connection and try again")
        return False, error_msg


def verify_model_cache(model_dir: str) -> bool:
    """
    Verify that all required models are present in cache.
    
    Quick check to ensure the model cache directory contains all necessary
    files for OPEA TTS synthesis without triggering downloads.
    
    Args:
        model_dir: Path to model cache directory
        
    Returns:
        True if all required models are cached, False otherwise
        
    Example:
        >>> if verify_model_cache("models/tts/opea_speecht5"):
        ...     print("Ready to synthesize!")
        ... else:
        ...     print("Need to download models first")
    """
    if not os.path.exists(model_dir):
        return False
    
    required_files = [
        "models--microsoft--speecht5_tts",
        "models--microsoft--speecht5_hifigan"
    ]
    
    return all(
        os.path.exists(os.path.join(model_dir, f))
        for f in required_files
    )
