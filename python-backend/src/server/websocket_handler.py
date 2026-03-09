"""
WebSocket Route Handler
=========================
Registers the /ws WebSocket endpoint on the FastAPI app and
routes incoming client messages to the appropriate handlers.
"""

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .connection_manager import ConnectionManager
from .pipeline_manager import PipelineManager
from .protocol import (
    ClientMessageType,
    parse_client_message,
    build_device_list,
    build_status_update,
    build_error,
    build_transcript_update,
    build_model_list,
    build_model_switched,
)


def create_websocket_route(
    app: FastAPI,
    manager: ConnectionManager,
    pipeline: PipelineManager,
) -> None:
    """
    Register the WebSocket endpoint and wire up the pipeline's broadcast.
    """
    # Give the pipeline a way to push messages to all connected clients
    pipeline.set_broadcast(manager.broadcast)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        """
        Handles a single WebSocket connection lifecycle:
        1. Accept the connection
        2. Send initial status + device list + model list
        3. Enter receive loop (route messages)
        4. Clean up on disconnect
        """
        await manager.connect(ws)

        try:
            # ── Send initial status on connect ──
            status = pipeline.get_status()
            await manager.send(ws, build_status_update(**status))

            # ── Send model list on connect ──
            try:
                models = pipeline.get_models_list()
                active_id = pipeline.active_model_id
                await manager.send(ws, build_model_list(models, active_id))
                print(f"[WS] Sent model list: {len(models)} model(s), active: {active_id}")
            except Exception as e:
                print(f"[WS] Failed to send model list: {e}")

            # ── Receive loop ──
            while True:
                raw = await ws.receive_text()
                msg_type, data = parse_client_message(raw)

                if msg_type is None:
                    await manager.send(
                        ws, build_error("Invalid message format")
                    )
                    continue

                await _handle_message(ws, manager, pipeline, msg_type, data)

        except WebSocketDisconnect:
            pass  # Normal disconnection
        except Exception as e:
            print(f"[WS] Unexpected connection error: {e}")
        finally:
            await manager.disconnect(ws)


async def _handle_message(
    ws: WebSocket,
    manager: ConnectionManager,
    pipeline: PipelineManager,
    msg_type: ClientMessageType,
    data: object,
) -> None:
    """Route a parsed client message to the appropriate handler."""

    # ── START PIPELINE ──────────────────────────
    if msg_type == ClientMessageType.START_PIPELINE:
        if pipeline.is_running or pipeline._starting:
            if pipeline.is_running:
                await manager.send(ws, build_error("Pipeline is already running"))
            else:
                await manager.send(ws, build_error("Pipeline is starting — please wait"))
            return

        try:
            await pipeline.start(data)
            status = pipeline.get_status()
            await manager.broadcast(build_status_update(**status))
            print("[WS] ▶ Pipeline started")
        except Exception as e:
            await manager.send(
                ws, build_error(f"Failed to start pipeline: {e}")
            )

    # ── STOP PIPELINE ───────────────────────────
    elif msg_type == ClientMessageType.STOP_PIPELINE:
        if not pipeline.is_running:
            await manager.send(ws, build_error("Pipeline is not running"))
            return

        try:
            await pipeline.stop()
            status = pipeline.get_status()
            await manager.broadcast(build_status_update(**status))
            print("[WS] ■ Pipeline stopped")
        except Exception as e:
            await manager.send(
                ws, build_error(f"Failed to stop pipeline: {e}")
            )

    # ── UPDATE SETTINGS ─────────────────────────
    elif msg_type == ClientMessageType.UPDATE_SETTINGS:
        try:
            pipeline.update_settings(data)
            status = pipeline.get_status()
            await manager.broadcast(build_status_update(**status))
        except Exception as e:
            await manager.send(
                ws, build_error(f"Failed to update settings: {e}")
            )

    # ── CLEAR TRANSCRIPT ────────────────────────
    elif msg_type == ClientMessageType.CLEAR_TRANSCRIPT:
        try:
            pipeline.clear_transcript()
            await manager.broadcast(
                build_transcript_update(
                    full_text="",
                    latest_word="",
                    is_sentence_complete=False,
                )
            )
            print("[WS] ✖ Transcript cleared")
        except Exception as e:
            await manager.send(
                ws, build_error(f"Failed to clear transcript: {e}")
            )

    # ── GET DEVICES ─────────────────────────────
    elif msg_type == ClientMessageType.GET_DEVICES:
        try:
            loop = asyncio.get_running_loop()
            cameras, audio_devices = await asyncio.gather(
                loop.run_in_executor(None, PipelineManager.enumerate_cameras),
                loop.run_in_executor(None, PipelineManager.enumerate_audio_devices),
            )

            await manager.send(ws, build_device_list(cameras, audio_devices))
            print(
                f"[WS] Sent device list: "
                f"{len(cameras)} camera(s), {len(audio_devices)} audio device(s)"
            )
        except Exception as e:
            await manager.send(
                ws, build_error(f"Failed to enumerate devices: {e}")
            )

    # ── GET MODELS ──────────────────────────────
    elif msg_type == ClientMessageType.GET_MODELS:
        try:
            models = pipeline.get_models_list()
            active_id = pipeline.active_model_id
            await manager.send(ws, build_model_list(models, active_id))
            print(f"[WS] Sent model list: {len(models)} model(s), active: {active_id}")
        except Exception as e:
            await manager.send(
                ws, build_error(f"Failed to get models: {e}")
            )

    # ── SWITCH MODEL ────────────────────────────
    elif msg_type == ClientMessageType.SWITCH_MODEL:
        try:
            model_id = data.get("model_id", "") if isinstance(data, dict) else str(data)
            if not model_id:
                await manager.send(ws, build_error("No model_id provided"))
                return

            print(f"[WS] Switching model to: {model_id}")
            result = await pipeline.switch_model(model_id)

            if result.get("success"):
                model_name = pipeline.active_model_name
                # Notify all clients about the switch
                await manager.broadcast(build_model_switched(model_id, model_name))
                # Send updated model list to all clients
                models = pipeline.get_models_list()
                await manager.broadcast(build_model_list(models, model_id))
                # Send updated status to all clients
                status = pipeline.get_status()
                await manager.broadcast(build_status_update(**status))
                print(f"[WS] ✓ Model switched to: {model_name} ({model_id})")
            else:
                error_msg = result.get("error", f"Failed to switch to model: {model_id}")
                await manager.send(ws, build_error(error_msg))
        except Exception as e:
            await manager.send(
                ws, build_error(f"Failed to switch model: {e}")
            )
