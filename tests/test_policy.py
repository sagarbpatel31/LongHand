"""CutPolicy: clean cut (§9.2), forced cut (§9.5), lead_pause, and invariants."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from longhand.segment.policy import CutPolicy
from longhand.segment.types import SegmentClosed
from longhand.segment.vad import FRAME_DURATION_S, frames_from_spans

RATE = 16000
TOL = FRAME_DURATION_S * 1.5  # boundary tolerance ~ one frame


class DummyRing:
    """Returns deterministically-sized bytes and records retain_from calls."""

    def __init__(self, rate: int = RATE) -> None:
        self.rate = rate
        self.retained: list[float] = []

    def slice(self, t_start: float, t_end: float) -> bytes:
        n = max(0, round((t_end - t_start) * self.rate))
        return b"\x00" * (n * 2)

    def retain_from(self, t: float) -> None:
        self.retained.append(t)


def _run(policy: CutPolicy, frames: list[bool]) -> list[SegmentClosed]:
    events: list[SegmentClosed] = []
    for i, sp in enumerate(frames):
        e = policy.on_frame(sp, i * FRAME_DURATION_S)
        if e is not None:
            events.append(e)
    tail = policy.flush(len(frames) * FRAME_DURATION_S)
    if tail is not None:
        events.append(tail)
    return events


def test_clean_cut_at_target_pause():
    # §9.2 — 8s speech, 0.7s pause, more speech -> a clean cut inside the pause.
    frames = frames_from_spans([("speech", 8.0), ("silence", 0.7), ("speech", 5.0)])
    events = _run(CutPolicy(ring=DummyRing()), frames)
    assert len(events) == 2
    assert events[0].seq == 0
    assert events[0].forced is False
    assert events[0].t_start == 0.0
    assert abs(events[0].t_end - 8.0) < 0.1  # cut at last speech, trailing silence trimmed
    assert abs(events[1].lead_pause - 0.7) < 0.1  # the pause becomes structure


def test_sub_target_pause_does_not_cut():
    frames = frames_from_spans([("speech", 8.0), ("silence", 0.4), ("speech", 8.0)])
    events = _run(CutPolicy(ring=DummyRing()), frames)
    assert len(events) == 1  # 0.4s < 0.6s target -> one segment
    assert events[0].forced is False


def test_below_min_segment_does_not_cut_on_pause():
    frames = frames_from_spans([("speech", 3.0), ("silence", 1.0), ("speech", 8.0)])
    events = _run(CutPolicy(ring=DummyRing()), frames)
    assert len(events) == 1  # 3s < 6s MIN_SEGMENT -> no cut at the pause
    assert events[0].t_start == 0.0


def test_forced_cut_at_max_segment():
    # §9.5 — 130s continuous speech, no pause -> forced cut + 2s overlap.
    ring = DummyRing()
    policy = CutPolicy(ring=ring)
    events = _run(policy, frames_from_spans([("speech", 130.0)]))
    assert len(events) == 2
    assert events[0].forced is True
    assert abs(events[0].t_end - 108.0) < 0.1  # MAX_SEGMENT content ceiling
    assert events[1].t_start == pytest.approx(106.0, abs=0.1)  # overlap prepended
    assert events[1].forced is False
    assert policy.forced_cut_rate == pytest.approx(0.5)
    assert ring.retained == pytest.approx([106.0], abs=0.1)


def test_lead_pause_recorded():
    frames = frames_from_spans([("silence", 3.0), ("speech", 8.0)])
    events = _run(CutPolicy(ring=DummyRing()), frames)
    assert len(events) == 1
    assert abs(events[0].lead_pause - 3.0) < 0.05


def test_seq_is_monotonic():
    frames = frames_from_spans(
        [("speech", 8.0), ("silence", 0.7), ("speech", 8.0), ("silence", 0.7), ("speech", 8.0)]
    )
    events = _run(CutPolicy(ring=DummyRing()), frames)
    assert [e.seq for e in events] == [0, 1, 2]


def test_widen_target_pause_reduces_segments():
    spans = [("speech", 8.0), ("silence", 0.7), ("speech", 8.0)]
    default = _run(CutPolicy(ring=DummyRing()), frames_from_spans(spans))

    widened = CutPolicy(ring=DummyRing())
    widened.widen_target_pause(1.0)  # 0.7s pause no longer qualifies
    widened_events = _run(widened, frames_from_spans(spans))

    assert len(widened_events) < len(default)
    assert len(default) == 2
    assert len(widened_events) == 1


def test_flush_closes_trailing_segment():
    events = _run(CutPolicy(ring=DummyRing()), frames_from_spans([("speech", 8.0)]))
    assert len(events) == 1
    assert events[0].forced is False
    assert events[0].t_start == 0.0


# --- property: segments ordered, non-overlapping (except forced), audio exact ---


@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.tuples(st.booleans(), st.integers(min_value=1, max_value=400)),
        min_size=1,
        max_size=8,
    )
)
def test_segments_ordered_and_cover_without_dropping(runs):
    frames: list[bool] = []
    for is_speech, n in runs:
        frames.extend([is_speech] * n)

    ring = DummyRing()
    policy = CutPolicy(ring=ring)
    events = _run(policy, frames)

    # seqs are contiguous from 0
    assert [e.seq for e in events] == list(range(len(events)))
    for i, e in enumerate(events):
        assert e.t_end > e.t_start  # non-empty
        # audio length matches the exact sliced range
        assert len(e.audio) == round((e.t_end - e.t_start) * RATE) * 2
        if i > 0:
            prev = events[i - 1]
            assert e.t_start >= prev.t_start  # ordered
            if not prev.forced:
                assert e.t_start >= prev.t_end - 1e-9  # clean cuts never overlap
