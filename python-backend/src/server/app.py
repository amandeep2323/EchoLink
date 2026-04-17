"""
FastAPI Application Factory
==============================
Creates and configures the FastAPI app with:
  - CORS middleware (allow frontend dev server)
  - Health check endpoint (GET /health)
  - WebSocket endpoint (ws://host:port/ws)
  - Lifespan management (startup/shutdown cleanup)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .connection_manager import ConnectionManager
from .pipeline_manager import PipelineManager
from .websocket_handler import create_websocket_route


# ── Module-level singletons ──
# These persist for the lifetime of the server process.
_manager = ConnectionManager()
_pipeline = PipelineManager()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    Lifespan context manager.
    - Startup:  log ready message
    - Shutdown: stop pipeline, close connections, release resources
    """
    print("[SignSpeak] Server ready — waiting for connections")
    yield
    print("[SignSpeak] Shutting down...")
    await _pipeline.cleanup()
    await _manager.close_all()
    print("[SignSpeak] Shutdown complete")


def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application."""

    app = FastAPI(
        title="SignSpeak Backend",
        description="ASL to Speech — WebSocket server for real-time sign language recognition",
        version="1.0.0",
        lifespan=_lifespan,
    )

    # ── CORS ──
    # Allow connections from the Vite dev server and any production origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health Check ──
    @app.get("/health")
    async def health():
        """
        Health check endpoint for monitoring.
        Returns server status, pipeline state, and connection count.
        """
        return {
            "status": "ok",
            "version": "1.0.0",
            "pipeline_running": _pipeline.is_running,
            "active_connections": _manager.active_count,
        }

    # ── WebSocket Route ──
    create_websocket_route(app, _manager, _pipeline)

    return app
