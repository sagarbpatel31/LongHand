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


def demo_fixture(*, n_clips: int = 42, words_per_clip: int = 34, gap_s: float = 1.5) -> Fixture:
    """A ~10-minute deterministic session for `python -m longhand.bench`."""
    clips = []
    k = 0
    for _ in range(n_clips):
        toks = [_VOCAB[(k + j) % len(_VOCAB)] for j in range(words_per_clip)]
        k += words_per_clip
        clips.append(" ".join(toks))
    return build_fixture(clips, gaps=gap_s)
