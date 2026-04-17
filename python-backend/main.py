"""
SignSpeak Backend — Entry Point
================================
Starts the FastAPI WebSocket server on port 8765.

Usage:
    python main.py

Endpoints:
    WebSocket:  ws://127.0.0.1:8765/ws
    Health:     http://127.0.0.1:8765/health
"""

import sys
import signal
import uvicorn
from src.server.app import create_app


def main():
    app = create_app()

    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=8765,
        log_level="info",
        ws_max_size=16 * 1024 * 1024,  # 16 MB max WebSocket message size
        timeout_keep_alive=30,
    )
    server = uvicorn.Server(config)

    # ── Graceful shutdown on SIGINT / SIGTERM ──
    shutdown_count = 0

    def handle_signal(sig, _frame):
        nonlocal shutdown_count
        shutdown_count += 1
        sig_name = signal.Signals(sig).name

        if shutdown_count == 1:
            print(f"\n[SignSpeak] Received {sig_name}, shutting down gracefully...")
            print("[SignSpeak] Press Ctrl+C again to force quit")
            server.should_exit = True
        elif shutdown_count >= 2:
            print(f"\n[SignSpeak] Force quit!")
            sys.exit(1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("=" * 60)
    print("  SignSpeak — ASL to Speech Backend")
    print("=" * 60)
    print(f"  WebSocket : ws://127.0.0.1:8765/ws")
    print(f"  Health    : http://127.0.0.1:8765/health")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("[SignSpeak] Server stopped")


if __name__ == "__main__":
    main()
