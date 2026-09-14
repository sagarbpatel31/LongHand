"""Benchmark harness — condition A (naive chop) vs B (VAD), through the real
scheduler + assembler. Offline, deterministic (FakeVad + fixture transcriber).

A: chop every `chop_s` seconds and concatenate — the baseline. Chops land inside
   words, so straddling words are lost at every boundary.
B: segment on silence via the VAD cut policy — cuts land in gaps, no word straddles
   a boundary, nothing is lost at the seam.

Condition C (glossary carryover + consistency pass) arrives in hours 7-9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..capture.ring import RingBuffer
from ..segment.policy import CutPolicy
from ..segment.types import SegmentClosed
from ..segment.vad import FRAME_DURATION_S, FakeVad
from ..stt.client import TranscriptionConfig
from ..stt.scheduler import transcribe_segments
from .fixture_client import FixtureDictationClient, words_lost
from .fixtures import Fixture
from .wer import wer

_BYTES_PER_SAMPLE = 2


@dataclass(frozen=True)
class ConditionResult:
    name: str
    n_segments: int
    wer: float
    boundary_word_loss: int
    forced_cut_rate: float
    hyp: str


def naive_chop_segments(
    fixture: Fixture, chop_s: float
) -> tuple[list[SegmentClosed], dict[int, tuple[float, float]]]:
    rate = fixture.sample_rate
    fb = _BYTES_PER_SAMPLE
    segs: list[SegmentClosed] = []
    ranges: dict[int, tuple[float, float]] = {}
    seq = 0
    t = 0.0
    while t < fixture.total_s - 1e-9:
        t0, t1 = t, min(t + chop_s, fixture.total_s)
        s0, s1 = round(t0 * rate) * fb, round(t1 * rate) * fb
        segs.append(
            SegmentClosed(seq=seq, audio=fixture.audio[s0:s1], t_start=t0, t_end=t1, lead_pause=0.0, forced=False)
        )
        ranges[seq] = (t0, t1)
        seq += 1
        t = t1
    return segs, ranges


def vad_segments(
    fixture: Fixture,
) -> tuple[list[SegmentClosed], dict[int, tuple[float, float]], float]:
    rate = fixture.sample_rate
    ring = RingBuffer(sample_rate=rate, max_seconds=fixture.total_s + 5.0)
    ring.write(np.frombuffer(fixture.audio, dtype=np.int16))
    vad = FakeVad(list(fixture.frame_script))
    policy = CutPolicy(ring=ring)

    segs: list[SegmentClosed] = []
    for i, sp in enumerate(fixture.frame_script):
        e = policy.on_frame(sp, i * FRAME_DURATION_S)
        if e is not None:
            segs.append(e)
    tail = policy.flush(len(fixture.frame_script) * FRAME_DURATION_S)
    if tail is not None:
        segs.append(tail)

    ranges = {s.seq: (s.t_start, s.t_end) for s in segs}
    return segs, ranges, policy.forced_cut_rate


async def _run(
    name: str,
    fixture: Fixture,
    segments: list[SegmentClosed],
    ranges: dict[int, tuple[float, float]],
    forced_rate: float,
) -> ConditionResult:
    client = FixtureDictationClient(fixture, ranges)
    base_cfg = TranscriptionConfig(sample_rate=fixture.sample_rate, channels=1, audio_format="pcm")
    doc = await transcribe_segments(segments, client, base_config=base_cfg, close_client=False)
    hyp = " ".join(s.best_text for s in doc if s.best_text)
    return ConditionResult(
        name=name,
        n_segments=len(segments),
        wer=wer(fixture.ground_truth, hyp),
        boundary_word_loss=words_lost(fixture, list(ranges.values())),
        forced_cut_rate=forced_rate,
        hyp=hyp,
    )


async def compare(fixture: Fixture, *, chop_s: float = 110.0) -> list[ConditionResult]:
    a_segs, a_ranges = naive_chop_segments(fixture, chop_s)
    b_segs, b_ranges, b_forced = vad_segments(fixture)
    a = await _run(f"A: naive {chop_s:.0f}s chop", fixture, a_segs, a_ranges, 0.0)
    b = await _run("B: VAD segmentation", fixture, b_segs, b_ranges, b_forced)
    return [a, b]


def format_table(results: list[ConditionResult], fixture: Fixture) -> str:
    n_words = len(fixture.words)
    avg = fixture.total_s / n_words if n_words else 0.0
    lines = [
        f"Session: {fixture.total_s:.1f}s, {n_words} words, {avg:.2f}s/word avg",
        "",
        f"{'Condition':<22}{'segs':>6}{'WER':>9}{'boundary loss':>15}{'forced%':>9}",
        "-" * 61,
    ]
    for r in results:
        lines.append(
            f"{r.name:<22}{r.n_segments:>6}{r.wer * 100:>8.1f}%"
            f"{r.boundary_word_loss:>15}{r.forced_cut_rate * 100:>8.1f}%"
        )
    if len(results) >= 2:
        a, b = results[0], results[1]
        delta = (a.wer - b.wer) * 100
        lines += ["", f"Headline: B cuts WER by {delta:.1f} points and saves "
                      f"{a.boundary_word_loss - b.boundary_word_loss} boundary words vs A."]
    return "\n".join(lines)
