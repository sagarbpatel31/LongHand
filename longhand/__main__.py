"""`python -m longhand` — serve the live-document UI.

Runs the FastAPI app (one HTML page over a WebSocket) that replays a synthetic
demo session through the real pipeline. No network, no mic, no API key needed.
"""

from __future__ import annotations


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn  # imported here so the offline test suite never needs uvicorn

    from .ui.app import create_app

    print(f"Longhand UI → http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
