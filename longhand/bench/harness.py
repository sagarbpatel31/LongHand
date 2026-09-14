"""Benchmark harness — condition A (naive chop) vs B (VAD), through the real
scheduler + assembler. Offline, deterministic (FakeVad + fixture transcriber).

A: chop every `chop_s` seconds and concatenate — the baseline. Chops land inside
   words, so straddling words are lost at every boundary.
B: segment on silence via the VAD cut policy — cuts land in gaps, no word straddles
   a boundary, nothing is lost at the seam.

Condition C (glossary carryover + consistency pass) arrives in hours 7-9.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..assemble.glossary import Glossary, consistency_pass
from ..capture.ring import RingBuffer
from ..segment.policy import CutPolicy
from ..segment.types import SegmentClosed
from ..segment.vad import FRAME_DURATION_S, FakeVad
from ..stt.client import TranscriptionConfig
from ..stt.scheduler import transcribe_segments
from .fixture_client import DriftDictationClient, FixtureDictationClient, words_lost
from .fixtures import DriftFixture, Fixture
from .metrics import (
    paragraph_boundary_f1,
    predicted_paragraph_times,
    terminology_consistency_rate,
)
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
    terminology_consistency: float | None = None
    paragraph_f1: float | None = None


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


async def _run_drift(
    name: str,
    drift: DriftFixture,
    segments: list[SegmentClosed],
    ranges: dict[int, tuple[float, float]],
    forced_rate: float,
    *,
    carryover: bool,
    c_in_flight: int = 1,
) -> ConditionResult:
    fx = drift.fixture
    base_cfg = TranscriptionConfig(sample_rate=fx.sample_rate, channels=1, audio_format="pcm")
    client = DriftDictationClient(drift, ranges)

    if carryover:
        glossary = Glossary()
        doc = await transcribe_segments(
            segments,
            client,
            base_config=base_cfg,
            close_client=False,
            max_in_flight=c_in_flight,  # serial -> deterministic causal carryover
            prepare_config=lambda seg: replace(
                base_cfg, seq=seg.seq, keyterms_prompt=glossary.keyterms() or None
            ),
            on_result=lambda seq, r: glossary.observe(r.best_text),
        )
    else:
        doc = await transcribe_segments(segments, client, base_config=base_cfg, close_client=False)

    hyp = " ".join(s.best_text for s in doc if s.best_text)
    if carryover:
        hyp = consistency_pass(hyp)  # deterministic safety net (§5.3)

    _, _, f1 = paragraph_boundary_f1(
        predicted_paragraph_times(doc, ranges), drift.paragraph_boundary_times
    )
    return ConditionResult(
        name=name,
        n_segments=len(segments),
        wer=wer(fx.ground_truth, hyp),
        boundary_word_loss=words_lost(fx, list(ranges.values())),
        forced_cut_rate=forced_rate,
        hyp=hyp,
        terminology_consistency=terminology_consistency_rate(hyp, drift),
        paragraph_f1=f1,
    )


async def compare_drift(
    drift: DriftFixture, *, chop_s: float = 110.0, c_in_flight: int = 1
) -> list[ConditionResult]:
    """A (naive chop) vs B (VAD) vs C (VAD + glossary carryover + consistency)."""
    fx = drift.fixture
    a_segs, a_ranges = naive_chop_segments(fx, chop_s)
    b_segs, b_ranges, b_forced = vad_segments(fx)
    a = await _run_drift(f"A: naive {chop_s:.0f}s chop", drift, a_segs, a_ranges, 0.0, carryover=False)
    b = await _run_drift("B: VAD, no carryover", drift, b_segs, b_ranges, b_forced, carryover=False)
    c = await _run_drift(
        "C: VAD + carryover", drift, b_segs, b_ranges, b_forced, carryover=True, c_in_flight=c_in_flight
    )
    return [a, b, c]


def format_table(results: list[ConditionResult], fixture: Fixture) -> str:
    n_words = len(fixture.words)
    avg = fixture.total_s / n_words if n_words else 0.0
    lines = [
        f"Session: {fixture.total_s:.1f}s, {n_words} words, {avg:.2f}s/word avg",
        "",
        f"{'Condition':<24}{'segs':>6}{'WER':>9}{'boundary loss':>15}{'forced%':>9}",
        "-" * 63,
    ]
    for r in results:
        lines.append(
            f"{r.name:<24}{r.n_segments:>6}{r.wer * 100:>8.1f}%"
            f"{r.boundary_word_loss:>15}{r.forced_cut_rate * 100:>8.1f}%"
        )

    if any(r.terminology_consistency is not None for r in results):
        lines += ["", "Terminology consistency (canonical / total rare-term occurrences)",
                  "-" * 63]
        for r in results:
            tc = r.terminology_consistency
            lines.append(f"{r.name:<24}{'n/a' if tc is None else f'{tc * 100:>6.1f}%':>9}")

    if any(r.paragraph_f1 is not None for r in results):
        lines += ["", "Paragraph-boundary F1 (structure recovered from pauses)", "-" * 63]
        for r in results:
            f1 = r.paragraph_f1
            lines.append(f"{r.name:<24}{'n/a' if f1 is None else f'{f1:>6.2f}':>9}")

    if len(results) >= 2:
        a, b = results[0], results[1]
        lines += ["", f"Headline: cutting at silence saves "
                      f"{a.boundary_word_loss - b.boundary_word_loss} boundary words vs naive chop."]
        if len(results) >= 3 and results[2].terminology_consistency is not None:
            c = results[2]
            lines.append(
                f"          carryover lifts terminology consistency "
                f"{b.terminology_consistency * 100:.0f}% -> {c.terminology_consistency * 100:.0f}%."
            )
    return "\n".join(lines)
