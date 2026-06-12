# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from opea-project/GenAIComps for desktop integration
# Original: https://github.com/opea-project/GenAIComps/tree/main/comps/tts
#
# Modifications:
#   - Removed Docker and microservice dependencies
#   - Added desktop-specific module structure
#   - Integrated with existing TTSEngine API

"""
Intel OPEA TTS Module

This module provides Intel OPEA's SpeechT5-based text-to-speech synthesis
as a self-contained desktop library without Docker/Kubernetes dependencies.

Public Interface:
    - OpeaTtsSynthesizer: TTSEngine-compatible interface for OPEA TTS

Usage:
    from opea_tts import OpeaTtsSynthesizer
    
    synthesizer = OpeaTtsSynthesizer(model_dir="models/tts/opea_speecht5")
    if synthesizer.load():
        audio = synthesizer.synthesize("Hello world")
"""

__version__ = "1.0.0"
__author__ = "Intel Corporation"

# Public interface exports
from .synthesizer import OpeaTtsSynthesizer
from .backend_detector import detect_inference_backend, check_openvino_available

__all__ = [
    "OpeaTtsSynthesizer",
    "detect_inference_backend",
    "check_openvino_available"
]
