"""LiveSession runs the real concurrent pipeline, offline (§2).

Drives synthetic frames through the genuine VAD+policy+scheduler+assembler path
with a `FakeVad` and a `FakeDictationClient` — no mic, no network. Proves the
live message protocol: init(mode=live), pending-before-document, ordered output
despite scrambled completion, a permanent failure surviving as a visible gap,
pauses becoming structure, and glossary carryover feeding later segments.
"""

from __future__ import annotations

import asyncio

import numpy as np

from longhand.segment.vad import FRAME_SIZE, FakeVad, frames_from_spans
from longhand.stt.client import ServerError
from longhand.stt.fake import FakeBehavior, FakeDictationClient, make_result
from longhand.ui.live import LiveSession

# Three speech runs; the 2.0s middle pause outranks the paragraph threshold (1.2s).
SPANS = [
    ("speech", 7.0),
    ("silence", 0.7),
    ("speech", 7.0),
    ("silence", 2.0),
    ("speech", 7.0),
]


async def _noop_sleep(_d: float) -> None:
    return None


def _frame(is_speech: bool) -> np.ndarray:
    amp = 8000 if is_speech else 0
    return np.full(FRAME_SIZE, amp, dtype=np.int16)


async def _run(session: LiveSession, script: list[bool]) -> list[dict]:
    async def drive() -> None:
        for sp in script:
            await session.feed_frame(_frame(sp))
        await session.finish_input()

    driver = asyncio.create_task(drive())
    msgs = [m async for m in session.stream()]
    await driver
    return msgs


def _blocks(doc: dict) -> list[dict]:
    return [b for s in doc["sections"] for p in s["paragraphs"] for b in p["blocks"]]


async def test_live_orders_scrambled_completion_and_carries_structure():
    script = frames_from_spans(SPANS)
    client = FakeDictationClient(
        sleep=_noop_sleep,
        behaviors={
            0: FakeBehavior(result=make_result("alpha"), latency=0.03),
            1: FakeBehavior(result=make_result("bravo"), latency=0.01),  # lands first
            2: FakeBehavior(result=make_result("charlie"), latency=0.02),
        },
    )
    session = LiveSession(FakeVad(script), client, close_client=False)
    msgs = await _run(session, script)

    init = msgs[0]
    assert init["type"] == "init" and init["mode"] == "live"
    assert init["n_segments"] is None  # unknown up front — it's live

    assert any(m["type"] == "pending" for m in msgs)
    docs = [m for m in msgs if m["type"] == "document"]
    final = docs[-1]
    assert final["final"] is True

    blocks = _blocks(final)
    assert [b["verbatim"] for b in blocks] == ["alpha", "bravo", "charlie"]  # ordered
    # live has no ground truth -> no WER / terminology score
    assert final["stats"]["wer"] is None
    assert final["stats"]["terminology_consistency"] is None
    # the 2.0s pause opened a second paragraph
    assert final["stats"]["paragraphs"] >= 2
    assert final["stats"]["segments_done"] == 3


async def test_live_permanent_failure_becomes_visible_gap():
    script = frames_from_spans(SPANS)
    client = FakeDictationClient(
        sleep=_noop_sleep,
        behaviors={
            0: FakeBehavior(result=make_result("alpha")),
            1: FakeBehavior(raises=ServerError("down")),  # never succeeds -> gap
            2: FakeBehavior(result=make_result("charlie")),
        },
    )
    session = LiveSession(FakeVad(script), client, close_client=False)
    final = [m for m in await _run(session, script) if m["type"] == "document"][-1]

    blocks = _blocks(final)
    assert len(blocks) == 3  # nothing dropped
    assert blocks[1]["is_gap"] and blocks[1]["verbatim"] == ""
    assert blocks[0]["verbatim"] == "alpha" and blocks[2]["verbatim"] == "charlie"
    assert final["stats"]["gaps"] == 1


async def test_live_pending_precedes_first_document():
    script = frames_from_spans(SPANS)
    session = LiveSession(
        FakeVad(script),
        FakeDictationClient(sleep=_noop_sleep, default=FakeBehavior(result=make_result("x"))),
        close_client=False,
        max_in_flight=1,  # serial -> deterministic ordering for the assertion
    )
    msgs = await _run(session, script)
    body = [m for m in msgs if m["type"] in ("pending", "document")]
    assert body[0]["type"] == "pending"  # a segment is announced before it lands


async def test_live_carryover_pins_terms_for_later_segments():
    # A glossary must learn from early results and feed keyterms into later config.
    # We assert the hook fires by observing keyterms grow across serial calls.
    script = frames_from_spans(SPANS)
    seen_keyterms: list[list[str] | None] = []

    class RecordingClient(FakeDictationClient):
        async def transcribe(self, audio, config):
            seen_keyterms.append(config.keyterms_prompt)
            return await super().transcribe(audio, config)

    client = RecordingClient(
        sleep=_noop_sleep,
        behaviors={
            0: FakeBehavior(result=make_result("kubernetes cluster reconciles state")),
            1: FakeBehavior(result=make_result("the kubernetes operator")),
            2: FakeBehavior(result=make_result("ingress routes traffic")),
        },
    )
    session = LiveSession(FakeVad(script), client, close_client=False, max_in_flight=1)
    await _run(session, script)

    assert seen_keyterms[0] is None  # first segment has nothing to carry
    # a later segment saw keyterms learned from earlier transcripts
    assert any(kt for kt in seen_keyterms[1:])
