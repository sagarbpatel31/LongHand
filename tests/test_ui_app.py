"""FastAPI app — in-process TestClient checks, no network, no live server (§2).

Confirms the page is served and the WebSocket streams the ReplaySession message
protocol end to end (init -> pending/document pairs -> a final scored document).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from longhand.ui.app import build_demo_drift, create_app
from longhand.ui.session import ReplaySession


def test_index_page_served():
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "Longhand" in r.text
    assert "/ws" in r.text  # the page knows how to open the socket


def test_websocket_streams_full_protocol():
    client = TestClient(create_app())
    with client.websocket_connect("/ws?latency=0") as ws:
        init = ws.receive_json()
        assert init["type"] == "init"
        assert init["n_segments"] > 0

        types: list[str] = []
        final = None
        for _ in range(4 * init["n_segments"] + 10):  # generous cap
            m = ws.receive_json()
            types.append(m["type"])
            if m["type"] == "document" and m["final"]:
                final = m
                break

        assert final is not None, "never received a final document"
        assert "pending" in types and "document" in types
        assert final["stats"]["segments_done"] == init["n_segments"]
        assert final["stats"]["wer"] is not None


def test_demo_session_has_exactly_one_forced_cut_for_the_seam_inspector():
    # The seam inspector needs something to inspect: the run-on monologue must
    # trip a forced cut while the structured clips still cut on silence.
    session = ReplaySession(build_demo_drift(), max_segment_s=30.0)
    forced = round(session.forced_cut_rate * session.n_segments)
    assert forced == 1
