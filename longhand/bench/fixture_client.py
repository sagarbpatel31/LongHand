"""A DictationClient that "transcribes" by slicing fixture ground truth (§9.1).

Models a *perfect* ASR so the benchmark isolates what the PIPELINE loses at
boundaries, not what the model gets wrong. Given a segment's [t_start, t_end]
(looked up by seq), it returns exactly the fixture words fully inside that range.
A word straddling a cut falls fully inside no segment and is therefore dropped —
that is the boundary word-loss condition A pays and condition B avoids.
"""

from __future__ import annotations

from ..stt.client import DictationClient, TranscriptionConfig, TranscriptionResult
from ..stt.fake import make_result
from .fixtures import DriftFixture, Fixture

# One-frame slack absorbs the VAD's 32ms quantization so clean (silence) cuts
# keep whole words; arbitrary mid-word chops still exceed it and drop the word.
from ..segment.vad import FRAME_DURATION_S

_EPS = FRAME_DURATION_S * 1.5


def word_fully_inside(w_start: float, w_end: float, t0: float, t1: float) -> bool:
    return w_start >= t0 - _EPS and w_end <= t1 + _EPS


def words_lost(fixture: Fixture, ranges: list[tuple[float, float]]) -> int:
    """Ground-truth words not fully contained in ANY single segment range."""
    lost = 0
    for w in fixture.words:
        if not any(word_fully_inside(w.t_start, w.t_end, t0, t1) for t0, t1 in ranges):
            lost += 1
    return lost


class FixtureDictationClient(DictationClient):
    def __init__(
        self,
        fixture: Fixture,
        ranges_by_seq: dict[int, tuple[float, float]],
    ) -> None:
        self._fixture = fixture
        self._ranges = ranges_by_seq

    async def transcribe(
        self, audio: bytes, config: TranscriptionConfig
    ) -> TranscriptionResult:
        t0, t1 = self._ranges[config.seq]
        tokens: list[tuple[float, str]] = []
        for w in self._fixture.words:
            if word_fully_inside(w.t_start, w.t_end, t0, t1):
                tokens.append((w.t_start, w.text))
            elif w.t_start < t1 and w.t_end > t0:
                # straddles this cut -> a mangled fragment (spec §0: "both adjacent
                # segments mangle the seam"). Won't match ground truth.
                tokens.append((w.t_start, w.text + "~"))
        tokens.sort()
        text = " ".join(tok for _, tok in tokens)
        return make_result(text, audio_duration_ms=round((t1 - t0) * 1000))


class DriftDictationClient(DictationClient):
    """Like FixtureDictationClient, but tagged rare terms drift on a minority of
    occurrences UNLESS pinned for that segment.

    Drift is periodic by the term's occurrence rank (deterministic, independent of
    processing order): rank 0 always lands correctly (bootstraps the glossary),
    every `drift_every`-th occurrence thereafter drifts. A term is *pinned* when its
    canonical form appears in the segment config's keyterms or stt_prompt — then it
    never drifts. Condition B supplies neither (drifts remain); condition C's
    glossary pins it after the first occurrence and the consistency pass cleans the
    residual.
    """

    def __init__(
        self,
        drift: DriftFixture,
        ranges_by_seq: dict[int, tuple[float, float]],
        *,
        drift_every: int = 3,
    ) -> None:
        self._drift = drift
        self._ranges = ranges_by_seq
        self._drift_every = drift_every
        self._rank: dict[int, tuple[str, int]] = {}
        for canon, positions in drift.canonical_positions.items():
            for r, idx in enumerate(positions):
                self._rank[idx] = (canon, r)

    def _pinned(self, config: TranscriptionConfig) -> set[str]:
        kt = config.keyterms_prompt or config.keyterms or config.word_boost or []
        prompt = (config.stt_prompt or config.prompt or "").lower()
        kt_lower = {t.lower() for t in kt}
        return {
            canon
            for canon in self._drift.drift_by_canonical
            if canon in kt_lower or canon in prompt
        }

    async def transcribe(
        self, audio: bytes, config: TranscriptionConfig
    ) -> TranscriptionResult:
        t0, t1 = self._ranges[config.seq]
        pins = self._pinned(config)
        fx = self._drift.fixture
        tokens: list[tuple[float, str]] = []
        for idx, w in enumerate(fx.words):
            if word_fully_inside(w.t_start, w.t_end, t0, t1):
                tagged = self._rank.get(idx)
                if tagged is not None:
                    canon, rank = tagged
                    drifts = rank % self._drift_every == self._drift_every - 1
                    if canon not in pins and drifts:
                        tokens.append((w.t_start, self._drift.drift_by_canonical[canon]))
                    else:
                        tokens.append((w.t_start, w.text))
                else:
                    tokens.append((w.t_start, w.text))
            elif w.t_start < t1 and w.t_end > t0:
                tokens.append((w.t_start, w.text + "~"))
        tokens.sort()
        return make_result(
            " ".join(tok for _, tok in tokens), audio_duration_ms=round((t1 - t0) * 1000)
        )
