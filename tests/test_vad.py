"""Vad interface: FakeVad + frames_from_spans. Silero is live-only (manual)."""

from __future__ import annotations

import numpy as np

from longhand.segment.vad import (
    FRAME_DURATION_S,
    FRAME_SIZE,
    FakeVad,
    frames_from_spans,
)

_SILENT = np.zeros(FRAME_SIZE, dtype=np.float32)


def test_fake_vad_returns_scripted_decisions():
    vad = FakeVad([True, True, False, True])
    assert [vad.is_speech(_SILENT) for _ in range(4)] == [True, True, False, True]


def test_fake_vad_past_end_is_silence():
    vad = FakeVad([True])
    assert vad.is_speech(_SILENT) is True
    assert vad.is_speech(_SILENT) is False  # past end


def test_fake_vad_reset_states_rewinds():
    vad = FakeVad([True, False])
    vad.is_speech(_SILENT)
    vad.reset_states()
    assert vad.is_speech(_SILENT) is True  # back to index 0


def test_fake_vad_callable_script():
    vad = FakeVad(lambda i: i % 2 == 0)
    assert [vad.is_speech(_SILENT) for _ in range(4)] == [True, False, True, False]


def test_speech_prob_default_from_is_speech():
    vad = FakeVad([True, False])
    assert vad.speech_prob(_SILENT) == 1.0
    assert vad.speech_prob(_SILENT) == 0.0


def test_frames_from_spans_lengths():
    frames = frames_from_spans([("speech", 8.0), ("silence", 0.7)])
    assert len(frames) == round(8.0 / FRAME_DURATION_S) + round(0.7 / FRAME_DURATION_S)
    assert all(frames[: round(8.0 / FRAME_DURATION_S)])  # speech block
    assert not any(frames[round(8.0 / FRAME_DURATION_S) :])  # silence block


def test_frames_from_spans_total_duration():
    frames = frames_from_spans([("speech", 3.2), ("silence", 1.6)])
    assert len(frames) == round(4.8 / FRAME_DURATION_S)
