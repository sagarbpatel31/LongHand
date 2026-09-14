"""splice.py: overlap merge (§6), property reconstruction (§9.3), forced cut (§9.5)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from longhand.assemble.splice import SEAM_MARKER, splice_texts, splice_words
from longhand.bench.fixture_client import word_fully_inside
from longhand.bench.fixtures import build_fixture
from longhand.capture.ring import RingBuffer
from longhand.segment.policy import CutPolicy
from longhand.segment.vad import FRAME_DURATION_S
from longhand.stt.client import Word


def _w(text: str, conf: float = 1.0) -> Word:
    return Word(text, conf)


# --- property: any split with overlap reconstructs the original (§9.3) -------


@settings(max_examples=200, deadline=None)
@given(st.integers(min_value=20, max_value=60), st.data())
def test_overlap_splice_reconstructs_original(n, data):
    tokens = [f"w{i}" for i in range(n)]  # distinct -> no coincidental matches
    ov = data.draw(st.integers(min_value=2, max_value=10))
    p = data.draw(st.integers(min_value=ov, max_value=n - 1))  # split point
    left = tokens[:p]
    right = tokens[p - ov :]  # begins with the ov-word overlap

    res = splice_words([_w(t) for t in left], [_w(t) for t in right])

    assert res.aligned
    assert res.overlap_len == ov
    assert [w.text for w in res.words] == tokens  # exact reconstruction, no dup


# --- confidence tie-break on disagreement (§6 step 3) -----------------------


def test_overlap_keeps_higher_confidence_word():
    left = [_w("the", 0.9), _w("dog", 0.3), _w("ran", 0.9)]
    right = [_w("the", 0.9), _w("dawg", 0.8), _w("ran", 0.9)]
    res = splice_words(left, right)  # 2/3 tokens match -> aligned at L=3
    assert res.aligned and res.overlap_len == 3
    assert [w.text for w in res.words] == ["the", "dawg", "ran"]  # dog(.3) -> dawg(.8)


# --- fallback: no alignment -> naive concat + visible marker (§6 step 4) -----


def test_no_overlap_falls_back_to_visible_seam():
    res = splice_texts("alpha beta", "gamma delta")
    assert res.aligned is False
    assert res.overlap_len == 0
    assert SEAM_MARKER in res.text
    assert [w.text for w in res.words] == ["alpha", "beta", "gamma", "delta"]


def test_below_threshold_falls_back():
    # only 1 of 3 overlap tokens match -> 0.33 < 0.6 -> no alignment
    left = [_w("aa"), _w("bb"), _w("cc")]
    right = [_w("cc"), _w("xx"), _w("yy")]
    res = splice_words(left, right)
    assert res.aligned is False


# --- integration: real forced cut -> overlapping transcripts -> splice -------


async def test_forced_cut_segments_splice_back_to_original():
    tokens = [f"w{i}" for i in range(300)]  # ~115s continuous -> a forced cut at 108s
    fx = build_fixture([" ".join(tokens)], gaps=0.0)
    ring = RingBuffer(sample_rate=fx.sample_rate, max_seconds=fx.total_s + 5.0)
    ring.write(np.frombuffer(fx.audio, dtype=np.int16))

    policy = CutPolicy(ring=ring)
    segs = []
    for i, sp in enumerate(fx.frame_script):
        e = policy.on_frame(sp, i * FRAME_DURATION_S)
        if e is not None:
            segs.append(e)
    tail = policy.flush(len(fx.frame_script) * FRAME_DURATION_S)
    if tail is not None:
        segs.append(tail)

    assert len(segs) == 2
    assert segs[0].forced is True  # first closed by MAX_SEGMENT
    assert segs[1].t_start < segs[0].t_end  # second carries the 2s overlap

    def clean_words(t0, t1):
        return [_w(w.text) for w in fx.words if word_fully_inside(w.t_start, w.t_end, t0, t1)]

    left = clean_words(segs[0].t_start, segs[0].t_end)
    right = clean_words(segs[1].t_start, segs[1].t_end)
    res = splice_words(left, right)

    assert res.aligned and res.overlap_len > 0
    assert [w.text for w in res.words] == tokens  # de-duplicated, nothing lost
