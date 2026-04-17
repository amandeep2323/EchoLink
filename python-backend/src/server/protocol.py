"""
WebSocket Protocol — Message Types & Builders
===============================================
Defines all message types exchanged between frontend and backend,
plus helper functions to build properly-formatted JSON messages.

Client → Server:
    start_pipeline, stop_pipeline, update_settings, get_devices,
    clear_transcript, get_models, switch_model

Server → Client:
    preview_frame, sign_detected, transcript_update, status_update,
    device_list, error, model_list, model_switched
"""

import json
from enum import Enum
from typing import Any, Optional


# ═══════════════════════════════════════════════
# Message Type Enums
# ═══════════════════════════════════════════════

class ClientMessageType(str, Enum):
    """Messages the frontend sends to the backend."""
    START_PIPELINE = "start_pipeline"
    STOP_PIPELINE = "stop_pipeline"
    UPDATE_SETTINGS = "update_settings"
    GET_DEVICES = "get_devices"
    CLEAR_TRANSCRIPT = "clear_transcript"
    GET_MODELS = "get_models"
    SWITCH_MODEL = "switch_model"


class ServerMessageType(str, Enum):
    """Messages the backend sends to the frontend."""
    PREVIEW_FRAME = "preview_frame"
    SIGN_DETECTED = "sign_detected"
    TRANSCRIPT_UPDATE = "transcript_update"
    STATUS_UPDATE = "status_update"
    DEVICE_LIST = "device_list"
    ERROR = "error"
    MODEL_LIST = "model_list"
    MODEL_SWITCHED = "model_switched"


# ═══════════════════════════════════════════════
# Message Builders (Server → Client)
# ═══════════════════════════════════════════════

def _build(msg_type: ServerMessageType, data: Any) -> str:
    """Build a JSON message string."""
    return json.dumps({"type": msg_type.value, "data": data}, separators=(",", ":"))


def build_preview_frame(frame_b64: str) -> str:
    """Base64 JPEG preview frame."""
    return _build(ServerMessageType.PREVIEW_FRAME, {"frame": frame_b64})


def build_sign_detected(
    sign: str,
    confidence: float,
    top_3: list[dict],
    letter_added: bool = False,
    smoothed_confidence: float = 0.0,
) -> str:
    """Sign detection result with top-3 predictions and acceptance status."""
    return _build(ServerMessageType.SIGN_DETECTED, {
        "sign": sign,
        "confidence": round(confidence, 4),
        "smoothed_confidence": round(smoothed_confidence, 4),
        "letter_added": letter_added,
        "top_3": [
            {"sign": p["sign"], "confidence": round(p["confidence"], 4)}
            for p in top_3
        ],
    })


def build_transcript_update(
    full_text: str,
    latest_word: str,
    is_sentence_complete: bool,
) -> str:
    """Transcript text update."""
    return _build(ServerMessageType.TRANSCRIPT_UPDATE, {
        "full_text": full_text,
        "latest_word": latest_word,
        "is_sentence_complete": is_sentence_complete,
    })


def build_status_update(
    pipeline_running: bool,
    model_loaded: bool,
    hands_detected: bool,
    vcam_active: bool,
    vmic_active: bool,
    fps: float,
    frames_processed: int,
    model_id: str = "",
    model_name: str = "",
    available_models: int = 0,
) -> str:
    """Pipeline status snapshot with model info."""
    return _build(ServerMessageType.STATUS_UPDATE, {
        "pipeline_running": pipeline_running,
        "model_loaded": model_loaded,
        "hands_detected": hands_detected,
        "vcam_active": vcam_active,
        "vmic_active": vmic_active,
        "fps": round(fps, 1),
        "frames_processed": frames_processed,
        "model_id": model_id,
        "model_name": model_name,
        "available_models": available_models,
    })


def build_device_list(
    cameras: list[dict],
    audio_output_devices: list[dict],
) -> str:
    """Available camera and audio output devices."""
    return _build(ServerMessageType.DEVICE_LIST, {
        "cameras": cameras,
        "audio_output_devices": audio_output_devices,
    })


def build_error(message: str) -> str:
    """Error notification."""
    return _build(ServerMessageType.ERROR, {"message": message})


def build_model_list(models: list[dict], active_model_id: str) -> str:
    """List of available models with active marker."""
    model_entries = []
    for m in models:
        model_entries.append({
            "id": m.get("id", ""),
            "name": m.get("name", "Unknown"),
            "description": m.get("description", ""),
            "type": m.get("type", "fingerspelling"),
            "active": m.get("id", "") == active_model_id,
            "model_type": m.get("model_type", "unknown"),
            "labels_count": m.get("labels_count", 0),
        })
    return _build(ServerMessageType.MODEL_LIST, {
        "models": model_entries,
        "active_model_id": active_model_id,
    })


def build_model_switched(model_id: str, model_name: str) -> str:
    """Confirmation that a model switch completed."""
    return _build(ServerMessageType.MODEL_SWITCHED, {
        "model_id": model_id,
        "model_name": model_name,
    })


# ═══════════════════════════════════════════════
# Message Parser (Client → Server)
# ═══════════════════════════════════════════════

def parse_client_message(raw: str) -> tuple[Optional[ClientMessageType], Any]:
    """
    Parse a raw JSON string from the client.
    Returns (message_type, data) or (None, None) on failure.
    """
    try:
        msg = json.loads(raw)
        msg_type = ClientMessageType(msg.get("type", ""))
        data = msg.get("data")
        return msg_type, data
    except (json.JSONDecodeError, ValueError, KeyError):
        return None, None
