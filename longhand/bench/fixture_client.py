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
from .fixtures import Fixture

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
