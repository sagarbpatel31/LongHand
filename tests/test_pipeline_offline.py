"""Hour-5 gate: synthetic audio -> segmenter -> scheduler -> assembler, offline.

No mic, no network, no fixtures. Proves the whole pipeline holds together:
ordered output despite scrambled completion, a permanent failure surviving as a
visible gap with neighbors intact, no audio dropped, and the concurrency cap held.
"""

from __future__ import annotations

import numpy as np

from longhand.assemble.order import OrderedAssembler
from longhand.capture.ring import RingBuffer
from longhand.segment.policy import CutPolicy
from longhand.segment.vad import FRAME_DURATION_S, frames_from_spans
from longhand.stt.client import ServerError, TranscriptionConfig
from longhand.stt.fake import FakeBehavior, FakeDictationClient, make_result
from longhand.stt.scheduler import SegmentScheduler

RATE = 16000
BASE_CFG = TranscriptionConfig(sample_rate=RATE, channels=1, audio_format="pcm")

# 4 speech segments; pauses vary (0.7s, 2.0s, 0.7s) so lead_pause carries structure.
SPANS = [
    ("speech", 7.0),
    ("silence", 0.7),
    ("speech", 7.0),
    ("silence", 2.0),
    ("speech", 7.0),
    ("silence", 0.7),
    ("speech", 7.0),
]


async def _noop_sleep(_d: float) -> None:
    return None


def _segment_via_policy():
    frames = frames_from_spans(SPANS)
    total_s = len(frames) * FRAME_DURATION_S
    ring = RingBuffer(sample_rate=RATE, max_seconds=total_s + 5.0)
    ring.write(np.zeros(round((total_s + 1.0) * RATE), dtype=np.int16))

    policy = CutPolicy(ring=ring)
    segments = []
    for i, sp in enumerate(frames):
        e = policy.on_frame(sp, i * FRAME_DURATION_S)
        if e is not None:
            segments.append(e)
    tail = policy.flush(total_s)
    if tail is not None:
        segments.append(tail)
    return segments


async def test_offline_pipeline_order_and_gaps():
    segments = _segment_via_policy()
    assert [s.seq for s in segments] == [0, 1, 2, 3]  # segmenter produced 4
    assert all(len(s.audio) > 0 for s in segments)  # audio never empty

    # Fake behaviors: seq1 fails permanently (-> gap); latencies scramble order.
    fake = FakeDictationClient(
        behaviors={
            0: FakeBehavior(result=make_result("seg0"), latency=0.03),
            1: FakeBehavior(raises=ServerError("down")),  # permanent
            2: FakeBehavior(result=make_result("seg2"), latency=0.01),  # completes first
            3: FakeBehavior(result=make_result("seg3"), latency=0.02),
        }
    )

    asm = OrderedAssembler()
    sched = SegmentScheduler(fake, base_config=BASE_CFG, assembler=asm, sleep=_noop_sleep)
    for seg in segments:
        await sched.submit(seg)
    await sched.close()
    await sched.run()
    doc = asm.emit_all()

    # ordered despite scrambled completion (§9.4)
    assert [s.seq for s in doc] == [0, 1, 2, 3]
    # no audio dropped — every segment is represented
    assert len(doc) == len(segments)
    # permanent failure is a visible gap, neighbors intact (§9.6)
    assert doc[1].is_gap and doc[1].best_text == ""
    assert doc[0].best_text == "seg0"
    assert doc[2].best_text == "seg2"
    assert doc[3].best_text == "seg3"
    # silence became structure: the 2.0s pause precedes seg2
    assert abs(doc[2].lead_pause - 2.0) < 0.1
    assert all(not s.forced for s in doc)
    # concurrency cap held
    assert sched.peak_in_flight <= 4
    # seq1 retried to exhaustion (retryable), others called once
    assert len([c for c in fake.calls if c.seq == 1]) == 3
