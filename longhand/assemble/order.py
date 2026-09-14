"""Out-of-order arrival -> ordered document.

`OrderedAssembler` accepts transcription results (or permanent-failure gap
markers) keyed by `seq`, holds gaps until the missing seqs arrive, and emits in
seq order. It is pure and synchronous — the scheduler is its only writer, so
there are no concurrency hazards here.

A permanently failed segment never disappears: it becomes a visible gap marker
that still advances the cursor, so it can neither block the segments after it nor
silently drop audio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..stt.client import TranscriptionResult

logger = logging.getLogger(__name__)


class AssemblyError(Exception):
    """Raised by emit_all() when the document still has holes."""


@dataclass(frozen=True)
class AssembledSegment:
    seq: int
    result: TranscriptionResult | None  # None iff this is a gap
    lead_pause: float
    forced: bool
    is_gap: bool
    error: str | None = None  # short reason for the gap marker (UI/debug)

    @property
    def best_text(self) -> str:
        """Cleaned/verbatim text, or "" for a gap (structure.py renders a marker)."""
        return self.result.best_text if self.result is not None else ""


class OrderedAssembler:
    def __init__(self, *, first_seq: int = 0) -> None:
        self._pending: dict[int, AssembledSegment] = {}
        self._cursor = first_seq  # next seq needed for a contiguous emit
        self._first_seq = first_seq
        self._max_seen = first_seq - 1
        self._emitted: list[AssembledSegment] = []

    # --- recording ----------------------------------------------------------

    def add_result(
        self,
        seq: int,
        result: TranscriptionResult,
        *,
        lead_pause: float,
        forced: bool,
    ) -> None:
        """Record a successful transcription. Idempotent per seq."""
        self._record(
            AssembledSegment(
                seq=seq,
                result=result,
                lead_pause=lead_pause,
                forced=forced,
                is_gap=False,
            )
        )

    def add_gap(
        self,
        seq: int,
        *,
        lead_pause: float,
        forced: bool,
        error: str,
    ) -> None:
        """Record a permanent failure as a visible gap. Never drops the seq."""
        logger.warning("seq=%s assemble.gap error=%s", seq, error)
        self._record(
            AssembledSegment(
                seq=seq,
                result=None,
                lead_pause=lead_pause,
                forced=forced,
                is_gap=True,
                error=error,
            )
        )

    def _record(self, seg: AssembledSegment) -> None:
        if seg.seq < self._cursor or seg.seq in self._pending:
            # Already emitted or already pending — idempotent no-op (defends
            # against a retry double-reporting the same seq).
            return
        self._pending[seg.seq] = seg
        self._max_seen = max(self._max_seen, seg.seq)

    # --- emission -----------------------------------------------------------

    def ready(self) -> list[AssembledSegment]:
        """Emit every contiguous element from the cursor, advancing it.

        Stops at the first hole. Safe to call repeatedly for streaming emit.
        """
        out: list[AssembledSegment] = []
        while self._cursor in self._pending:
            seg = self._pending.pop(self._cursor)
            out.append(seg)
            self._emitted.append(seg)
            self._cursor += 1
        return out

    def emit_all(self) -> list[AssembledSegment]:
        """Final assembly. Raises if any seq below max_seen is still missing."""
        self.ready()
        if self._pending:
            missing = sorted(
                s
                for s in range(self._first_seq, self._max_seen + 1)
                if s >= self._cursor and s not in self._pending
            )
            raise AssemblyError(f"document has holes; missing seqs: {missing}")
        return list(self._emitted)

    def document(self) -> list[AssembledSegment]:
        """Snapshot of everything emitted so far (contiguous, in seq order)."""
        self.ready()
        return list(self._emitted)
