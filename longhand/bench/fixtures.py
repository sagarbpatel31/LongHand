"""Synthetic long-session fixtures with exact ground truth (§8).

Recording and hand-transcribing 10 minutes is expensive. Instead we place words
of a fixed duration on a timeline, separate clips with known silence gaps, and
render matching int16 audio (tone during words, silence during gaps) plus a
per-frame speech/silence script for `FakeVad`.

Because we control the timeline exactly, we get: (a) exact ground truth for WER,
(b) known gaps to segment on, and (c) known word boundaries so condition A's
mid-word chops are measurable. No network, no mic, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..segment.vad import FRAME_DURATION_S

SAMPLE_RATE = 16000
_AMPLITUDE = 8000  # nonzero tone during speech


@dataclass(frozen=True)
class TimedWord:
    text: str
    t_start: float
    t_end: float


@dataclass(frozen=True)
class Fixture:
    words: tuple[TimedWord, ...]
    total_s: float
    audio: bytes  # int16 LE, 16kHz mono
    frame_script: tuple[bool, ...]  # per 32ms frame, for FakeVad
    sample_rate: int = SAMPLE_RATE

    @property
    def ground_truth(self) -> str:
        return " ".join(w.text for w in self.words)


def build_fixture(
    clips: list[str],
    *,
    gaps: list[float] | float = 1.5,
    word_s: float = 0.384,  # 12 frames @ 32ms — aligns cleanly to the VAD grid
    sample_rate: int = SAMPLE_RATE,
) -> Fixture:
    """Build a session from `clips` (space-separated words) joined by `gaps`.

    `gaps[i]` is the silence after clip i (the last gap is ignored/trailing). A
    scalar applies one gap between every clip. Words within a clip are contiguous.
    """
    n_clips = len(clips)
    gap_list = gaps if isinstance(gaps, list) else [gaps] * n_clips

    words: list[TimedWord] = []
    speech_spans: list[tuple[float, float]] = []
    t = 0.0
    for ci, clip in enumerate(clips):
        for token in clip.split():
            ws, we = t, t + word_s
            words.append(TimedWord(token, ws, we))
            speech_spans.append((ws, we))
            t = we
        if ci < n_clips:  # trailing gap after each clip
            t += gap_list[ci] if ci < len(gap_list) else 0.0
    total_s = t

    n_samples = round(total_s * sample_rate)
    audio = np.zeros(n_samples, dtype=np.int16)
    for ws, we in speech_spans:
        s0, s1 = round(ws * sample_rate), round(we * sample_rate)
        audio[s0:s1] = _AMPLITUDE

    n_frames = int(total_s / FRAME_DURATION_S)
    frame_script: list[bool] = []
    for i in range(n_frames):
        t_mid = (i + 0.5) * FRAME_DURATION_S
        frame_script.append(any(ws <= t_mid < we for ws, we in speech_spans))

    return Fixture(
        words=tuple(words),
        total_s=total_s,
        audio=audio.tobytes(),
        frame_script=tuple(frame_script),
        sample_rate=sample_rate,
    )


# A small deterministic vocabulary for generating demo sessions without randomness.
_VOCAB = (
    "the kubernetes cluster reconciles desired state through a control loop "
    "operators watch resources and drive them toward the manifest we declared "
    "meanwhile the scheduler places pods onto nodes with available capacity "
    "and the ingress routes external traffic to the backing services"
).split()


def _demo_clips(n_clips: int, words_per_clip: int) -> list[str]:
    clips = []
    k = 0
    for _ in range(n_clips):
        clips.append(" ".join(_VOCAB[(k + j) % len(_VOCAB)] for j in range(words_per_clip)))
        k += words_per_clip
    return clips


def demo_fixture(*, n_clips: int = 42, words_per_clip: int = 34, gap_s: float = 1.5) -> Fixture:
    """A ~10-minute deterministic session for `python -m longhand.bench`."""
    return build_fixture(_demo_clips(n_clips, words_per_clip), gaps=gap_s)


# --- drift-tagged fixtures (condition C benchmark) --------------------------


@dataclass(frozen=True)
class DriftTerm:
    canonical: str  # e.g. "kubernetes"
    drift: str  # e.g. "koobernetes" — the mis-transcribed variant


@dataclass(frozen=True)
class DriftFixture:
    fixture: Fixture
    canonical_positions: dict[str, tuple[int, ...]]  # canon(lower) -> word indices
    drift_by_canonical: dict[str, str]  # canon(lower) -> drift variant
    paragraph_boundary_times: tuple[float, ...]  # ground-truth paragraph breaks (session-time)

    @property
    def ground_truth(self) -> str:
        return self.fixture.ground_truth


def build_drift_fixture(
    clips: list[str],
    *,
    gaps: list[float] | float = 1.5,
    drift_terms: list[DriftTerm],
    word_s: float = 0.384,
    sample_rate: int = SAMPLE_RATE,
    paragraph_pause: float = 1.2,
) -> DriftFixture:
    """A `Fixture` plus tagged rare-term positions and ground-truth paragraph
    boundaries. Tagging is metadata only — the audio/timeline are unchanged."""
    fixture = build_fixture(clips, gaps=gaps, word_s=word_s, sample_rate=sample_rate)
    drift_by = {dt.canonical.lower(): dt.drift for dt in drift_terms}

    positions: dict[str, list[int]] = {k: [] for k in drift_by}
    for i, w in enumerate(fixture.words):
        c = w.text.lower()
        if c in drift_by:
            positions[c].append(i)

    gap_list = gaps if isinstance(gaps, list) else [gaps] * len(clips)
    t = 0.0
    clip_start: list[float] = []
    for ci, clip in enumerate(clips):
        clip_start.append(t)
        t += len(clip.split()) * word_s
        t += gap_list[ci] if ci < len(gap_list) else 0.0
    para_times = tuple(
        clip_start[i]
        for i in range(1, len(clips))
        if (gap_list[i - 1] if i - 1 < len(gap_list) else 0.0) >= paragraph_pause
    )

    return DriftFixture(
        fixture=fixture,
        canonical_positions={k: tuple(v) for k, v in positions.items()},
        drift_by_canonical=drift_by,
        paragraph_boundary_times=para_times,
    )


_DEMO_DRIFT_TERMS = [
    DriftTerm("kubernetes", "koobernetes"),
    DriftTerm("ingress", "ingres"),
    DriftTerm("scheduler", "schedular"),
]


def demo_drift_fixture(*, n_clips: int = 42, words_per_clip: int = 34) -> DriftFixture:
    """~10-min drift-tagged session with varied gaps (sentence/paragraph/section)."""
    gaps = [[0.8, 1.5, 4.0][i % 3] for i in range(n_clips)]
    return build_drift_fixture(
        _demo_clips(n_clips, words_per_clip), gaps=gaps, drift_terms=_DEMO_DRIFT_TERMS
    )
