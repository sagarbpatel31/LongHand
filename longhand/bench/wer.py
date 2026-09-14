"""Word error rate — token-level edit distance."""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> list[str]:
    return _PUNCT.sub("", text.lower()).split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance over token sequences."""
    m, n = len(ref), len(hyp)
    if m == 0:
        return n
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate = edit_distance / len(reference tokens)."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)
