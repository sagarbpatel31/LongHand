"""Benchmark metrics beyond WER: terminology consistency + paragraph-boundary F1."""

from __future__ import annotations

from collections import Counter

from ..assemble.order import AssembledSegment
from ..assemble.structure import build_document
from .fixtures import DriftFixture
from .wer import normalize


def terminology_consistency_rate(doc_text: str, drift: DriftFixture) -> float:
    """Fraction of tagged rare-term occurrences that appear in canonical form."""
    counts = Counter(normalize(doc_text))
    canon_hits = drift_hits = 0
    for canon, variant in drift.drift_by_canonical.items():
        canon_hits += counts[canon]
        drift_hits += counts[variant.lower()]
    denom = canon_hits + drift_hits
    return 1.0 if denom == 0 else canon_hits / denom


def predicted_paragraph_times(
    doc: list[AssembledSegment],
    ranges_by_seq: dict[int, tuple[float, float]],
    *,
    paragraph_pause: float = 1.2,
    section_pause: float = 3.0,
) -> list[float]:
    """Session-times at which structure.build_document opens a new paragraph."""
    document = build_document(doc, paragraph_pause=paragraph_pause, section_pause=section_pause)
    return [ranges_by_seq[doc[i].seq][0] for i in document.paragraph_boundaries]


def paragraph_boundary_f1(
    pred_times, true_times, *, tol: float = 0.25
) -> tuple[float, float, float]:
    """Precision/recall/F1 of predicted paragraph boundaries vs ground truth,
    matched by session-time within `tol` seconds."""
    pred = sorted(pred_times)
    true = sorted(true_times)
    if not pred and not true:
        return (1.0, 1.0, 1.0)

    used = [False] * len(true)
    matched = 0
    for p in pred:
        for j, t in enumerate(true):
            if not used[j] and abs(p - t) <= tol:
                used[j] = True
                matched += 1
                break

    precision = matched / len(pred) if pred else (1.0 if not true else 0.0)
    recall = matched / len(true) if true else (1.0 if not pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return (precision, recall, f1)
