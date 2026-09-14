"""LiveSession — the real pipeline, live, over a WebSocket (§2).

Where `ReplaySession` replays a synthetic fixture serially, `LiveSession` runs
the genuine concurrent pipeline as audio arrives: frames -> VAD -> cut policy ->
parallel scheduler -> ordered assembler -> glossary carryover, emitting the SAME
`init`/`pending`/`document` messages so the one HTML page renders it unchanged.

Frames enter through a single junction, `on_segment`, regardless of source:
- live: `capture.mic.MicCapture` runs VAD + policy on the audio thread and calls
  `on_segment` via `run_coroutine_threadsafe`;
- offline tests: `feed_frame()` runs the identical VAD + policy step in-loop with
  a `FakeVad` and a fake client — no mic, no network, deterministic.

Results land out of order; each landing emits a fresh document snapshot of the
contiguous prefix the assembler has emitted so far. The whole thing is driven by
one `asyncio.Queue`: `on_segment` enqueues `pending`, `on_result` enqueues a
`result` marker, and `stream()` is the sole consumer that turns markers into
messages — so the assembler is only ever touched from the single consumer path.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import AsyncIterator

import numpy as np

from ..assemble.glossary import Glossary
from ..assemble.order import OrderedAssembler
from ..assemble.structure import PARAGRAPH_PAUSE_S, SECTION_PAUSE_S
from ..capture.ring import RingBuffer
from ..segment.policy import MAX_SEGMENT_S, CutPolicy
from ..segment.types import SegmentClosed
from ..segment.vad import FRAME_DURATION_S, FRAME_SIZE, Vad
from ..stt.client import DictationClient, TranscriptionConfig, TranscriptionResult
from ..stt.scheduler import MAX_IN_FLIGHT, SegmentScheduler
from .messages import document_message, pending_message

logger = logging.getLogger(__name__)


class LiveSession:
    def __init__(
        self,
        vad: Vad,
        client: DictationClient,
        *,
        sample_rate: int = 16000,
        max_segment_s: float = MAX_SEGMENT_S,
        max_in_flight: int = MAX_IN_FLIGHT,
        max_seconds: float = 1800.0,  # ring capacity (30 min) — bounds memory
        close_client: bool = True,
    ) -> None:
        self.ring = RingBuffer(sample_rate=sample_rate, max_seconds=max_seconds)
        self.vad = vad
        self.policy = CutPolicy(ring=self.ring, max_segment_s=max_segment_s)

        self._sample_rate = sample_rate
        self._base_cfg = TranscriptionConfig(
            sample_rate=sample_rate, channels=1, audio_format="pcm"
        )
        self._glossary = Glossary()
        self._asm = OrderedAssembler()
        self._sched = SegmentScheduler(
            client,
            base_config=self._base_cfg,
            assembler=self._asm,
            max_in_flight=max_in_flight,
            prepare_config=self._prepare_config,
            on_result=self._on_result,
            close_client=close_client,
        )
        self._q: asyncio.Queue = asyncio.Queue()
        self._frame_index = 0
        self._submitted = 0
        self._input_closed = False

    # --- carryover hooks (run on the scheduler's task path) -----------------

    def _prepare_config(self, seg: SegmentClosed) -> TranscriptionConfig:
        return replace(
            self._base_cfg, seq=seg.seq, keyterms_prompt=self._glossary.keyterms() or None
        )

    def _on_result(self, seq: int, result: TranscriptionResult) -> None:
        self._glossary.observe(result.best_text)  # best-effort live carryover
        self._q.put_nowait(("result", seq))

    # --- the single frame junction ------------------------------------------

    async def on_segment(self, seg: SegmentClosed) -> None:
        """Called by the mic thread (bridged) or by feed_frame() for one segment."""
        self._submitted += 1
        self._q.put_nowait(("pending", seg))
        logger.debug("seq=%s live.submit forced=%s", seg.seq, seg.forced)
        await self._sched.submit(seg)

    async def feed_frame(self, frame_i16: np.ndarray) -> None:
        """Offline driver: one int16 frame (FRAME_SIZE samples) through VAD+policy.

        Mirrors `MicCapture._callback` minus sounddevice/threading.
        """
        if frame_i16.dtype != np.int16 or frame_i16.shape[-1] != FRAME_SIZE:
            raise ValueError(f"expected {FRAME_SIZE} int16 samples, got {frame_i16.shape} {frame_i16.dtype}")
        self.ring.write(frame_i16)
        f = frame_i16.astype(np.float32) / 32768.0
        t = self._frame_index * FRAME_DURATION_S
        self._frame_index += 1
        event = self.policy.on_frame(self.vad.is_speech(f), t)
        if event is not None:
            await self.on_segment(event)

    async def finish_input(self) -> None:
        """Offline driver: flush the tail (our own frame clock) then drain.

        Only valid on the `feed_frame` path, which owns `_frame_index`. The mic
        path advances the policy on the audio thread and flushes there, so it
        calls `close_input()` instead.
        """
        if self._input_closed:
            return
        event = self.policy.flush(self._frame_index * FRAME_DURATION_S)
        if event is not None:
            await self.on_segment(event)
        await self.close_input()

    async def close_input(self) -> None:
        """No more segments will be submitted: let the scheduler drain. Idempotent."""
        if self._input_closed:
            return
        self._input_closed = True
        await self._sched.close()

    # --- the message stream (sole assembler reader) -------------------------

    async def stream(self) -> AsyncIterator[dict]:
        yield self._init_msg()

        run_task = asyncio.create_task(self._sched.run())
        run_task.add_done_callback(lambda _t: self._q.put_nowait(("done",)))

        while True:
            item = await self._q.get()
            kind = item[0]
            if kind == "done":
                break
            if kind == "pending":
                done = len(self._asm.document())
                yield pending_message(item[1], done, self._submitted)
            elif kind == "result":
                assembled = self._asm.document()  # contiguous prefix emitted so far
                yield document_message(
                    assembled, len(assembled), self._submitted, final=False
                )

        assembled = self._asm.document()
        yield document_message(assembled, len(assembled), self._submitted, final=True)
        await run_task  # surface any scheduler error

    def _init_msg(self) -> dict:
        return {
            "type": "init",
            "mode": "live",
            "sample_rate": self._sample_rate,
            "n_segments": None,  # unknown up front — this is live
            "paragraph_pause_s": PARAGRAPH_PAUSE_S,
            "section_pause_s": SECTION_PAUSE_S,
            "drift_terms": [],
            "paragraph_boundary_times": [],
        }
