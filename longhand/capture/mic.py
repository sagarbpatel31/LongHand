"""Mic capture — sounddevice RawInputStream -> ring + VAD + cut policy.

Live-only, NOT unit-tested (tests never touch a real mic). `sounddevice` is
imported lazily inside `start()` so importing this module in CI without PortAudio
never fails.

The audio callback runs on sounddevice's thread; it must stay non-blocking (no
awaits, no heavy work). When the policy closes a segment we bridge it to the
asyncio event loop via `run_coroutine_threadsafe` so the scheduler can pick it up.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import numpy as np

from ..segment.policy import CutPolicy
from ..segment.types import SegmentClosed
from ..segment.vad import FRAME_DURATION_S, FRAME_SIZE, Vad
from .ring import RingBuffer

logger = logging.getLogger(__name__)


class MicCapture:
    def __init__(
        self,
        ring: RingBuffer,
        vad: Vad,
        policy: CutPolicy,
        on_segment: Callable[[SegmentClosed], Awaitable[None]],
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self._ring = ring
        self._vad = vad
        self._policy = policy
        self._on_segment = on_segment
        self._loop = loop
        self._sample_rate = sample_rate
        self._channels = channels
        self._stream = None
        self._leftover = np.empty(0, dtype=np.float32)  # samples not yet framed
        self._frame_index = 0  # complete VAD frames consumed == session clock

    def start(self) -> None:
        import sounddevice as sd  # lazy: PortAudio not needed for import/tests

        self._loop = self._loop or asyncio.get_event_loop()
        self._vad.reset_states()
        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=FRAME_SIZE,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("mic.start rate=%d ch=%d", self._sample_rate, self._channels)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # don't drop trailing audio
        event = self._policy.flush(self._frame_index * FRAME_DURATION_S)
        if event is not None:
            self._dispatch(event)
        logger.info("mic.stop frames=%d", self._frame_index)

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.warning("mic.callback status=%s", status)
        data = bytes(indata)
        self._ring.write(data)  # every sample is retained for slicing/retry

        block = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        self._leftover = np.concatenate((self._leftover, block))
        while len(self._leftover) >= FRAME_SIZE:
            frame = self._leftover[:FRAME_SIZE]
            self._leftover = self._leftover[FRAME_SIZE:]
            t = self._frame_index * FRAME_DURATION_S
            self._frame_index += 1
            event = self._policy.on_frame(self._vad.is_speech(frame), t)
            if event is not None:
                self._dispatch(event)

    def _dispatch(self, event: SegmentClosed) -> None:
        assert self._loop is not None
        asyncio.run_coroutine_threadsafe(self._on_segment(event), self._loop)
