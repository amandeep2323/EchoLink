# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from opea-project/GenAIComps for desktop integration
# Original: https://github.com/opea-project/GenAIComps/tree/main/comps/tts
#
# Backend detector for runtime OpenVINO/PyTorch detection

"""
Backend Detection Module

Detects available inference backends at runtime:
- OpenVINO: Intel's inference optimization toolkit (preferred when available)
- PyTorch CPU: Fallback inference engine

This module enables graceful degradation when OpenVINO is not installed.
"""


def detect_inference_backend() -> str:
    """
    Detect available inference backend at runtime.
    
    This function checks for OpenVINO availability and returns the appropriate
    backend identifier for the OPEA TTS module. If OpenVINO is not available,
    it falls back to PyTorch CPU inference.
    
    Returns:
        str: Backend identifier:
            - "openvino" if OpenVINO is available
            - "cpu" if only PyTorch CPU is available (fallback)
    
    Side Effects:
        Prints detection status to console for diagnostic purposes.
    
    Example:
        >>> backend = detect_inference_backend()
        [OPEA TTS] OpenVINO detected: 2023.1.0
        >>> print(backend)
        'openvino'
    """
    try:
        import openvino
        print(f"[OPEA TTS] OpenVINO detected: {openvino.__version__}")
        return "openvino"
    except ImportError:
        print("[OPEA TTS] OpenVINO not available — using PyTorch CPU")
        return "cpu"


def check_openvino_available() -> bool:
    """
    Check if OpenVINO is installed and importable.
    
    This is a lightweight check used by the TTSEngine to determine whether
    to prioritize the OPEA backend in "auto" mode. Unlike detect_inference_backend(),
    this function does not print diagnostic messages.
    
    Returns:
        bool: True if OpenVINO package is available, False otherwise
    
    Example:
        >>> if check_openvino_available():
        ...     print("OpenVINO acceleration is available")
        ...     # Prioritize OPEA TTS backend
        ... else:
        ...     print("Using PyTorch CPU fallback")
    """
    try:
        import openvino
        return True
    except ImportError:
        return False
