"""Forced-cut seam splicing (§6) — the rare path.

When a segment is force-closed at MAX_SEGMENT, the next segment is prepended with
the last FORCED_OVERLAP seconds of audio, so both transcribe the overlap region.
Concatenating naively would duplicate those words. Here we find the overlap by
aligning the tail of the left transcript against the head of the right, keep the
overlap once (choosing the higher-confidence word where they disagree), and drop
the duplication.

If no alignment scores above threshold we fall back to naive concatenation with a
visible seam marker — an honest visible seam beats a silent duplication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..stt.client import Word

SEAM_MARKER = "[⚠ seam]"

MAX_WINDOW = 15  # compare at most the last/first N words (§6: "~15 words")
MIN_OVERLAP = 2  # avoid single-word coincidental splices
THRESHOLD = 0.6  # min fraction of matching tokens to accept an alignment


@dataclass(frozen=True)
class SpliceResult:
    words: list[Word]
    text: str
    aligned: bool  # False -> fell back to naive concat + visible marker
    overlap_len: int


def _norm(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _best_overlap(
    left_norm: list[str],
    right_norm: list[str],
    *,
    max_window: int,
    min_overlap: int,
    threshold: float,
) -> tuple[int, float]:
    """Largest overlap length L (<= window) whose left-suffix ~ right-prefix
    match fraction is >= threshold. Returns (L, score); (0, 0.0) if none."""
    max_l = min(len(left_norm), len(right_norm), max_window)
    for length in range(max_l, min_overlap - 1, -1):
        matches = sum(
            1
            for a, b in zip(left_norm[-length:], right_norm[:length])
            if a == b and a != ""
        )
        if matches / length >= threshold:
            return length, matches / length
    return 0, 0.0


def splice_words(
    left: list[Word],
    right: list[Word],
    *,
    max_window: int = MAX_WINDOW,
    min_overlap: int = MIN_OVERLAP,
    threshold: float = THRESHOLD,
) -> SpliceResult:
    """Merge two overlapping forced-cut transcripts into one, de-duplicated."""
    left_norm = [_norm(w.text) for w in left]
    right_norm = [_norm(w.text) for w in right]
    length, _score = _best_overlap(
        left_norm, right_norm, max_window=max_window, min_overlap=min_overlap, threshold=threshold
    )

    if length == 0:
        words = list(left) + list(right)
        text = " ".join(
            [w.text for w in left] + [SEAM_MARKER] + [w.text for w in right]
        )
        return SpliceResult(words=words, text=text, aligned=False, overlap_len=0)

    # Keep the overlap once; on disagreement keep the higher-confidence word.
    merged = [
        a if a.confidence >= b.confidence else b
        for a, b in zip(left[len(left) - length :], right[:length])
    ]
    words = list(left[: len(left) - length]) + merged + list(right[length:])
    text = " ".join(w.text for w in words)
    return SpliceResult(words=words, text=text, aligned=True, overlap_len=length)


def splice_texts(left_text: str, right_text: str, **kw) -> SpliceResult:
    """Convenience for when only text is available (unit-confidence words)."""
    left = [Word(t, 1.0) for t in left_text.split()]
    right = [Word(t, 1.0) for t in right_text.split()]
    return splice_words(left, right, **kw)
