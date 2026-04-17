"""
WebSocket Connection Manager
==============================
Tracks active WebSocket connections. Provides thread-safe
send (single client) and broadcast (all clients) methods
with automatic cleanup of dead connections.
"""

import asyncio
from fastapi import WebSocket


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    # ── Properties ──────────────────────────────

    @property
    def active_count(self) -> int:
        """Number of currently connected clients."""
        return len(self._connections)

    # ── Connection Lifecycle ────────────────────

    async def connect(self, ws: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        print(f"[WS] Client connected  (active: {self.active_count})")

    async def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self._connections.discard(ws)
        print(f"[WS] Client disconnected (active: {self.active_count})")

    # ── Messaging ───────────────────────────────

    async def send(self, ws: WebSocket, message: str) -> None:
        """Send a message to a single client. Disconnects on failure."""
        try:
            await ws.send_text(message)
        except Exception:
            await self.disconnect(ws)

    async def broadcast(self, message: str) -> None:
        """
        Send a message to ALL connected clients.
        Dead connections are automatically cleaned up.
        """
        # Snapshot connections under lock
        async with self._lock:
            targets = list(self._connections)

        if not targets:
            return

        # Send to all; collect failures
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)

        # Clean up dead connections
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)
            print(f"[WS] Cleaned up {len(dead)} dead connection(s)")

    async def close_all(self) -> None:
        """Close all active connections (used during shutdown)."""
        async with self._lock:
            connections = list(self._connections)
            self._connections.clear()

        for ws in connections:
            try:
                await ws.close()
            except Exception:
                pass

        if connections:
            print(f"[WS] Closed {len(connections)} connection(s) on shutdown")
