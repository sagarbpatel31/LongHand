"""ui — FastAPI + one HTML page over a WebSocket (§2).

`app.create_app()` serves the live-document page; `session.ReplaySession` replays
a synthetic drift session through the real pipeline. `python -m longhand` runs it.
Offline and deterministic — no network, no mic.
"""
