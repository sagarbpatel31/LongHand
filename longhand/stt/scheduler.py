"""Async segment scheduler — parallel in-flight, retries, backpressure.

Consumes `SegmentClosed` events, transcribes them through a `DictationClient`
with a bounded concurrency cap, retries retryable failures from the retained
in-memory audio, and feeds results into an `OrderedAssembler` so the document
comes out in seq order regardless of completion order.

Buffered upload (decided): each attempt re-sends the full `seg.audio` bytes with
a Content-Length, so a retry is always replayable — there is no chunked-mid-body
failure mode. A permanently failed segment becomes a visible gap; audio is never
silently dropped.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Awaitable, Callable, Iterable

from ..assemble.order import AssembledSegment, OrderedAssembler
from ..segment.types import SegmentClosed
from .client import (
    DictationClient,
    DictationError,
    TranscriptionConfig,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)

MAX_IN_FLIGHT = 4
MAX_RETRIES = 2  # up to 3 attempts total
BASE_BACKOFF_S = 0.5
BACKPRESSURE_QUEUE_HIGH = 8


class SegmentScheduler:
    def __init__(
        self,
        client: DictationClient,
        *,
        base_config: TranscriptionConfig,
        assembler: OrderedAssembler | None = None,
        max_in_flight: int = MAX_IN_FLIGHT,
        max_retries: int = MAX_RETRIES,
        base_backoff_s: float = BASE_BACKOFF_S,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_backpressure: Callable[[int], None] | None = None,
        backpressure_high: int = BACKPRESSURE_QUEUE_HIGH,
        close_client: bool = True,
        prepare_config: Callable[[SegmentClosed], TranscriptionConfig] | None = None,
        on_result: Callable[[int, TranscriptionResult], None] | None = None,
    ) -> None:
        self._client = client
        self._base_config = base_config
        self._assembler = assembler or OrderedAssembler()
        # Carryover hooks (§5). Defaults reproduce the plain per-seq config path.
        self._prepare_config = prepare_config or (
            lambda seg: replace(self._base_config, seq=seg.seq)
        )
        self._on_result = on_result
        self._sem = asyncio.Semaphore(max_in_flight)
        self._max_retries = max_retries
        self._base_backoff_s = base_backoff_s
        self._sleep = sleep  # injected so backoff is deterministic in tests
        self._on_backpressure = on_backpressure
        self._backpressure_high = backpressure_high
        self._close_client = close_client

        self._tasks: set[asyncio.Task] = set()
        self._queue_depth = 0  # submitted but not yet started (waiting on the cap)
        self._in_flight = 0
        self._peak_in_flight = 0
        self._closed = asyncio.Event()
        self._aclosed = False

    # --- public API ---------------------------------------------------------

    @property
    def assembler(self) -> OrderedAssembler:
        return self._assembler

    @property
    def peak_in_flight(self) -> int:
        return self._peak_in_flight

    async def submit(self, seg: SegmentClosed) -> None:
        """Enqueue one closed segment for transcription (non-blocking wrt API)."""
        self._queue_depth += 1
        if self._on_backpressure is not None and self._queue_depth > self._backpressure_high:
            # Signal the segmenter to widen TARGET_PAUSE (fewer, longer segments).
            self._on_backpressure(self._queue_depth)
        task = asyncio.create_task(self._worker(seg))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self) -> None:
        """Signal that no more segments will be submitted."""
        self._closed.set()

    async def run(self) -> OrderedAssembler:
        """Drain until close() and all in-flight work + retries finish."""
        try:
            await self._closed.wait()
            pending = list(self._tasks)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        finally:
            await self.aclose()
        return self._assembler

    async def aclose(self) -> None:
        """Cancel in-flight tasks and close the client. Idempotent (§12)."""
        if self._aclosed:
            return
        self._aclosed = True
        self._closed.set()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
        if self._close_client:
            await self._client.aclose()

    # --- internals ----------------------------------------------------------

    async def _worker(self, seg: SegmentClosed) -> None:
        async with self._sem:
            self._queue_depth -= 1
            self._in_flight += 1
            self._peak_in_flight = max(self._peak_in_flight, self._in_flight)
            try:
                await self._attempt(seg)
            finally:
                self._in_flight -= 1

    async def _attempt(self, seg: SegmentClosed) -> None:
        # prepare_config runs now (at cap-slot acquisition), so a carryover hook
        # sees the most-recent-available transcripts (§5.2 graceful degradation).
        cfg = self._prepare_config(seg)
        if cfg.seq != seg.seq:
            cfg = replace(cfg, seq=seg.seq)  # seq is load-bearing; never drop it
        attempt = 0
        while True:
            try:
                result = await self._client.transcribe(seg.audio, cfg)
                self._assembler.add_result(
                    seg.seq, result, lead_pause=seg.lead_pause, forced=seg.forced
                )
                if self._on_result is not None:
                    self._on_result(seg.seq, result)
                logger.debug("seq=%s scheduler.ok attempt=%d", seg.seq, attempt)
                return
            except DictationError as e:
                retryable = getattr(e, "retryable", False)
                if retryable and attempt < self._max_retries:
                    delay = self._base_backoff_s * (2**attempt)
                    logger.info(
                        "seq=%s scheduler.retry attempt=%d after=%.2fs err=%s",
                        seg.seq,
                        attempt + 1,
                        delay,
                        type(e).__name__,
                    )
                    attempt += 1
                    await self._sleep(delay)
                    continue
                self._gap(seg, type(e).__name__)
                return
            except ValueError as e:
                # Client's own pre-network guard (e.g. over-length). Non-retryable.
                logger.warning("seq=%s scheduler.rejected %s", seg.seq, e)
                self._gap(seg, "ValueError")
                return

    def _gap(self, seg: SegmentClosed, error: str) -> None:
        self._assembler.add_gap(
            seg.seq, lead_pause=seg.lead_pause, forced=seg.forced, error=error
        )


async def transcribe_segments(
    segments: Iterable[SegmentClosed],
    client: DictationClient,
    *,
    base_config: TranscriptionConfig,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_backpressure: Callable[[int], None] | None = None,
    close_client: bool = True,
    max_in_flight: int = MAX_IN_FLIGHT,
    prepare_config: Callable[[SegmentClosed], TranscriptionConfig] | None = None,
    on_result: Callable[[int, TranscriptionResult], None] | None = None,
) -> list[AssembledSegment]:
    """One-shot: feed all segments, drain, return the ordered document.

    Used by the offline pipeline test and the bench harness.
    """
    asm = OrderedAssembler()
    sched = SegmentScheduler(
        client,
        base_config=base_config,
        assembler=asm,
        sleep=sleep,
        on_backpressure=on_backpressure,
        close_client=close_client,
        max_in_flight=max_in_flight,
        prepare_config=prepare_config,
        on_result=on_result,
    )
    for seg in segments:
        await sched.submit(seg)
    await sched.close()
    await sched.run()
    return asm.emit_all()
