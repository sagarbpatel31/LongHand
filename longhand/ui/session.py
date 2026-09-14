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

from ..assemble.glossary import Glossary
from ..assemble.order import OrderedAssembler
from ..assemble.structure import PARAGRAPH_PAUSE_S, SECTION_PAUSE_S
from ..bench.fixture_client import DriftDictationClient
from ..bench.fixtures import DriftFixture
from ..capture.ring import RingBuffer
from ..segment.policy import MAX_SEGMENT_S, CutPolicy
from ..segment.types import SegmentClosed
from ..segment.vad import FRAME_DURATION_S
from ..stt.client import TranscriptionConfig
from .messages import clean_blocks as _clean_blocks  # re-export for tests
from .messages import document_message, pending_message


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
            yield pending_message(seg, done, total)
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
            yield document_message(
                asm.document(), done + 1, total, final=is_last, drift=self._drift
            )

    # --- init message (replay-specific: drift terms + ground-truth pauses) --

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
