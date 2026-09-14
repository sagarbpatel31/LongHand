"""ReplaySession — drives the whole condition-C pipeline for the UI (§2, hours 10-12).

The UI never touches the network. Instead this replays a drift-tagged fixture
through the *real* pipeline — VAD segmentation, the drift ASR client, glossary
carryover, ordered assembly, pause->structure, and forced-cut splicing — and
emits a stream of JSON-serialisable messages a single HTML page renders live:

    init      once, up front: session shape + drift terms + pause thresholds.
    pending   before each segment transcribes: drives the in-flight shimmer.
    document  after each segment lands: a full snapshot of the growing document
              (sections -> paragraphs -> blocks), each block carrying BOTH the
              verbatim ASR text and the consistency-cleaned text (verbatim toggle),
              plus seam/splice info on any block that follows a forced cut.

Serial by design (one segment at a time): the carryover is causal, so a segment
sees the glossary built from the segments before it — same ordering the bench's
condition C uses (`max_in_flight=1`). Deterministic; no clocks, no randomness.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import AsyncIterator, Awaitable, Callable

import numpy as np

from ..assemble.glossary import Glossary, consistency_pass
from ..assemble.order import AssembledSegment, OrderedAssembler
from ..assemble.splice import splice_words
from ..assemble.structure import (
    PARAGRAPH_PAUSE_S,
    SECTION_PAUSE_S,
    Block,
    build_document,
)
from ..bench.fixture_client import DriftDictationClient
from ..bench.fixtures import DriftFixture
from ..bench.metrics import terminology_consistency_rate
from ..bench.wer import wer
from ..segment.policy import MAX_SEGMENT_S, CutPolicy
from ..segment.types import SegmentClosed
from ..segment.vad import FRAME_DURATION_S
from ..capture.ring import RingBuffer
from ..stt.client import TranscriptionConfig


async def _anoop(_delay: float) -> None:  # default pace: instant (tests)
    return None


def _segment(
    fixture, *, max_segment_s: float
) -> tuple[list[SegmentClosed], dict[int, tuple[float, float]], float]:
    """Run the fixture's frame script through the real cut policy (offline)."""
    rate = fixture.sample_rate
    ring = RingBuffer(sample_rate=rate, max_seconds=fixture.total_s + 5.0)
    ring.write(np.frombuffer(fixture.audio, dtype=np.int16))
    policy = CutPolicy(ring=ring, max_segment_s=max_segment_s)

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


class ReplaySession:
    """Replays one drift fixture through condition C, emitting UI messages."""

    def __init__(
        self,
        drift: DriftFixture,
        *,
        max_segment_s: float = MAX_SEGMENT_S,
        latency_s: float = 0.0,
        sleep: Callable[[float], Awaitable[None]] = None,
    ) -> None:
        self._drift = drift
        self._fixture = drift.fixture
        self._segs, self._ranges, self._forced_rate = _segment(
            self._fixture, max_segment_s=max_segment_s
        )
        self._client = DriftDictationClient(drift, self._ranges)
        self._latency_s = latency_s
        self._sleep = sleep or _anoop

    # --- introspection (app/tests) -----------------------------------------

    @property
    def n_segments(self) -> int:
        return len(self._segs)

    @property
    def forced_cut_rate(self) -> float:
        return self._forced_rate

    # --- the message stream -------------------------------------------------

    async def stream(self) -> AsyncIterator[dict]:
        """Yield init, then (pending, document) for each segment in order."""
        yield self._init_msg()

        base_cfg = TranscriptionConfig(
            sample_rate=self._fixture.sample_rate, channels=1, audio_format="pcm"
        )
        glossary = Glossary()
        asm = OrderedAssembler()
        total = len(self._segs)

        for done, seg in enumerate(self._segs):
            yield self._pending_msg(seg, done, total)
            await self._sleep(self._latency_s)

            cfg = replace(
                base_cfg, seq=seg.seq, keyterms_prompt=glossary.keyterms() or None
            )
            result = await self._client.transcribe(seg.audio, cfg)
            glossary.observe(result.best_text)  # causal carryover for the next seg
            asm.add_result(
                seg.seq, result, lead_pause=seg.lead_pause, forced=seg.forced
            )
            asm.ready()

            is_last = done + 1 == total
            yield self._document_msg(asm.document(), done + 1, total, final=is_last)

    # --- message builders ---------------------------------------------------

    def _init_msg(self) -> dict:
        return {
            "type": "init",
            "total_s": round(self._fixture.total_s, 3),
            "sample_rate": self._fixture.sample_rate,
            "n_segments": len(self._segs),
            "forced_cut_rate": round(self._forced_rate, 4),
            "paragraph_pause_s": PARAGRAPH_PAUSE_S,
            "section_pause_s": SECTION_PAUSE_S,
            "drift_terms": [
                {"canonical": canon, "drift": drift}
                for canon, drift in sorted(self._drift.drift_by_canonical.items())
            ],
            "paragraph_boundary_times": [
                round(t, 3) for t in self._drift.paragraph_boundary_times
            ],
        }

    def _pending_msg(self, seg: SegmentClosed, done: int, total: int) -> dict:
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

    def _document_msg(
        self,
        assembled: list[AssembledSegment],
        done: int,
        total: int,
        *,
        final: bool,
    ) -> dict:
        verbatims = [a.best_text for a in assembled]
        cleans = _clean_blocks(verbatims)
        by_seq = {a.seq: a for a in assembled}
        clean_by_seq = {a.seq: c for a, c in zip(assembled, cleans)}
        prev_by_seq = {
            assembled[i].seq: (assembled[i - 1] if i > 0 else None)
            for i in range(len(assembled))
        }

        doc = build_document(assembled)
        sections_out = []
        for section in doc.sections:
            paras_out = []
            for para in section.paragraphs:
                blocks_out = [
                    self._block_msg(
                        b, by_seq[b.seq], prev_by_seq[b.seq], clean_by_seq[b.seq]
                    )
                    for b in para.blocks
                ]
                paras_out.append({"blocks": blocks_out})
            sections_out.append({"paragraphs": paras_out})

        clean_full = " ".join(c for c in cleans if c).strip()
        tc = terminology_consistency_rate(clean_full, self._drift) if final else None
        stats = {
            "segments_done": done,
            "segments_total": total,
            "sections": len(doc.sections),
            "paragraphs": sum(len(s.paragraphs) for s in doc.sections),
            "forced_cuts": sum(1 for a in assembled if a.forced),
            "gaps": sum(1 for a in assembled if a.is_gap),
            "terminology_consistency": None if tc is None else round(tc, 4),
            "wer": round(wer(self._drift.ground_truth, clean_full), 4) if final else None,
        }
        return {
            "type": "document",
            "seq": assembled[-1].seq if assembled else None,
            "sections": sections_out,
            "stats": stats,
            "final": final,
        }

    def _block_msg(
        self,
        block: Block,
        seg: AssembledSegment,
        prev: AssembledSegment | None,
        clean: str,
    ) -> dict:
        return {
            "seq": block.seq,
            "verbatim": block.text,
            "clean": clean,
            "is_gap": block.is_gap,
            "forced": block.forced,
            "error": block.error,
            "lead_pause": round(block.lead_pause, 3),
            "seam": _seam_for(prev, seg),
        }


def _clean_blocks(verbatims: list[str]) -> list[str]:
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


def _seam_for(prev: AssembledSegment | None, cur: AssembledSegment) -> dict | None:
    """Splice info for `cur` iff the segment before it was a forced cut.

    The forced segment's tail overlaps `cur`'s head (§6). We align + dedupe, and
    expose both the naive (duplicated) concat and the spliced result so the seam
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
