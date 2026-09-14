"""Shared UI message builders (§2).

Both the offline `ReplaySession` and the live `LiveSession` emit the SAME
`pending` / `document` message shapes so one HTML page renders either source.
The only difference is scoring: a replay knows its ground truth (WER + terminology
consistency), a live session does not — pass `drift=None` and those fields are
null while everything else (structure, verbatim/clean, seams) is identical.
"""

from __future__ import annotations

from ..assemble.glossary import consistency_pass
from ..assemble.order import AssembledSegment
from ..assemble.splice import splice_words
from ..assemble.structure import Block, build_document
from ..bench.fixtures import DriftFixture
from ..bench.metrics import terminology_consistency_rate
from ..bench.wer import wer
from ..segment.types import SegmentClosed


def clean_blocks(verbatims: list[str]) -> list[str]:
    """Consistency-normalise the whole document, then re-split per block.

    `consistency_pass` rewrites tokens in place and never adds/removes them, so
    the cleaned token stream re-splits by each block's original word count. Gaps
    contribute zero tokens (empty verbatim) and stay empty.
    """
    joined = " ".join(v for v in verbatims)
    cleaned = consistency_pass(joined) if joined.strip() else joined
    ctoks = cleaned.split()
    out: list[str] = []
    i = 0
    for v in verbatims:
        n = len(v.split())
        out.append(" ".join(ctoks[i : i + n]))
        i += n
    return out


def seam_for(prev: AssembledSegment | None, cur: AssembledSegment) -> dict | None:
    """Splice info for `cur` iff the segment before it was a forced cut (§6).

    The forced segment's tail overlaps `cur`'s head. We align + dedupe and expose
    both the naive (duplicated) concat and the spliced result so the seam
    inspector can show exactly what was removed.
    """
    if prev is None or not prev.forced:
        return None
    if prev.result is None or cur.result is None:  # a gap on either side
        return None
    sr = splice_words(prev.result.words, cur.result.words)
    raw_concat = (prev.best_text + " " + cur.best_text).strip()
    return {
        "aligned": sr.aligned,
        "overlap_len": sr.overlap_len,
        "raw_concat": raw_concat,
        "spliced": sr.text,
        "marker": not sr.aligned,
    }


def pending_message(seg: SegmentClosed, done: int, total: int) -> dict:
    return {
        "type": "pending",
        "seq": seg.seq,
        "t_start": round(seg.t_start, 3),
        "t_end": round(seg.t_end, 3),
        "duration_s": round(seg.t_end - seg.t_start, 3),
        "lead_pause": round(seg.lead_pause, 3),
        "forced": seg.forced,
        "done": done,
        "total": total,
    }


def _block_message(block: Block, seg: AssembledSegment, prev: AssembledSegment | None, clean: str) -> dict:
    return {
        "seq": block.seq,
        "verbatim": block.text,
        "clean": clean,
        "is_gap": block.is_gap,
        "forced": block.forced,
        "error": block.error,
        "lead_pause": round(block.lead_pause, 3),
        "seam": seam_for(prev, seg),
    }


def document_message(
    assembled: list[AssembledSegment],
    done: int,
    total: int,
    *,
    final: bool,
    drift: DriftFixture | None = None,
) -> dict:
    """A full snapshot of the document so far: sections -> paragraphs -> blocks."""
    verbatims = [a.best_text for a in assembled]
    cleans = clean_blocks(verbatims)
    by_seq = {a.seq: a for a in assembled}
    clean_by_seq = {a.seq: c for a, c in zip(assembled, cleans)}
    prev_by_seq = {
        assembled[i].seq: (assembled[i - 1] if i > 0 else None)
        for i in range(len(assembled))
    }

    doc = build_document(assembled)
    sections_out = [
        {
            "paragraphs": [
                {
                    "blocks": [
                        _block_message(
                            b, by_seq[b.seq], prev_by_seq[b.seq], clean_by_seq[b.seq]
                        )
                        for b in para.blocks
                    ]
                }
                for para in section.paragraphs
            ]
        }
        for section in doc.sections
    ]

    clean_full = " ".join(c for c in cleans if c).strip()
    # Scores need ground truth; only a replay (drift fixture) has it.
    tc = terminology_consistency_rate(clean_full, drift) if (final and drift) else None
    wer_v = round(wer(drift.ground_truth, clean_full), 4) if (final and drift) else None

    stats = {
        "segments_done": done,
        "segments_total": total,
        "sections": len(doc.sections),
        "paragraphs": sum(len(s.paragraphs) for s in doc.sections),
        "forced_cuts": sum(1 for a in assembled if a.forced),
        "gaps": sum(1 for a in assembled if a.is_gap),
        "terminology_consistency": None if tc is None else round(tc, 4),
        "wer": wer_v,
    }
    return {
        "type": "document",
        "seq": assembled[-1].seq if assembled else None,
        "sections": sections_out,
        "stats": stats,
        "final": final,
    }
