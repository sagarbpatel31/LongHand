"""Cut policy — turns per-frame VAD decisions into `SegmentClosed` events.

Pure state machine over `(is_speech, timestamp)` frames. It never holds audio
itself: it asks the ring buffer for the bytes of each closed segment, so it stays
trivially testable and the memory bound lives in one place.

The two ideas of the project live here:
- **Cut at silence, not the cap.** A clean cut lands inside a pause (nothing to
  dedup); a forced cut is the rare fallback when a speaker runs past MAX_SEGMENT.
- **Silence is structure.** `lead_pause` (silence before each segment) is recorded
  and carried through — `structure.py` turns it into paragraph/section breaks.
"""

from __future__ import annotations

import logging
from typing import Protocol

from .types import SegmentClosed

logger = logging.getLogger(__name__)

MIN_SEGMENT_S = 6.0
TARGET_PAUSE_S = 0.600
# Content ceiling < client.MAX_SEGMENT_SECONDS (110). With FORCED_OVERLAP prepended
# to the next segment, audio handed to transcribe() stays safely under 110s.
MAX_SEGMENT_S = 108.0
FORCED_OVERLAP_S = 2.0


class _Ring(Protocol):
    def slice(self, t_start: float, t_end: float) -> bytes: ...
    def retain_from(self, t: float) -> None: ...


class CutPolicy:
    def __init__(
        self,
        *,
        ring: _Ring,
        min_segment_s: float = MIN_SEGMENT_S,
        target_pause_s: float = TARGET_PAUSE_S,
        max_segment_s: float = MAX_SEGMENT_S,
        forced_overlap_s: float = FORCED_OVERLAP_S,
    ) -> None:
        self._ring = ring
        self._min_segment_s = min_segment_s
        self._target_pause_s = target_pause_s
        self._max_segment_s = max_segment_s
        self._forced_overlap_s = forced_overlap_s

        self._seq = 0
        self._seg_open = False
        self._seg_start_t = 0.0  # audio slice start (may include forced overlap)
        self._seg_content_start_t = 0.0  # where real speech content began
        self._last_speech_t = 0.0
        self._prev_end_t = 0.0  # t_end of the previous segment (for lead_pause)
        self._cur_lead_pause = 0.0
        self._forced_count = 0
        self._total_count = 0

    @property
    def forced_cut_rate(self) -> float:
        return self._forced_count / self._total_count if self._total_count else 0.0

    def widen_target_pause(self, new_target_s: float) -> None:
        """Backpressure hook: fewer, longer segments when the queue backs up."""
        logger.info("policy.widen_target_pause %.3f -> %.3f", self._target_pause_s, new_target_s)
        self._target_pause_s = new_target_s

    def on_frame(self, is_speech: bool, t: float) -> SegmentClosed | None:
        if is_speech:
            if not self._seg_open:
                self._open(t)
            self._last_speech_t = t
            if (t - self._seg_start_t) >= self._max_segment_s:
                return self._close(t, forced=True)
            return None

        # silence
        if self._seg_open:
            if (t - self._seg_start_t) >= self._max_segment_s:
                return self._close(t, forced=True)
            silence_run = t - self._last_speech_t
            content_len = self._last_speech_t - self._seg_content_start_t
            if silence_run >= self._target_pause_s and content_len >= self._min_segment_s:
                return self._close(self._last_speech_t, forced=False)
        return None

    def flush(self, t: float) -> SegmentClosed | None:
        """End of stream: close any open segment (never drop trailing audio)."""
        if not self._seg_open:
            return None
        if self._last_speech_t <= self._seg_content_start_t:
            # only the reopened overlap, no new speech — nothing to emit
            self._seg_open = False
            return None
        return self._close(self._last_speech_t, forced=False)

    # --- internals ----------------------------------------------------------

    def _open(self, t: float) -> None:
        self._seg_open = True
        self._seg_start_t = t
        self._seg_content_start_t = t
        self._cur_lead_pause = max(0.0, t - self._prev_end_t)
        self._last_speech_t = t

    def _close(self, end_t: float, *, forced: bool) -> SegmentClosed:
        seq = self._seq
        audio = self._ring.slice(self._seg_start_t, end_t)
        event = SegmentClosed(
            seq=seq,
            audio=audio,
            t_start=self._seg_start_t,
            t_end=end_t,
            lead_pause=self._cur_lead_pause,
            forced=forced,
        )
        logger.debug(
            "seq=%s policy.cut forced=%s t_start=%.3f t_end=%.3f lead_pause=%.3f",
            seq,
            forced,
            self._seg_start_t,
            end_t,
            self._cur_lead_pause,
        )
        self._seq += 1
        self._total_count += 1
        self._prev_end_t = end_t
        self._seg_open = False

        if forced:
            self._forced_count += 1
            overlap_start = max(0.0, end_t - self._forced_overlap_s)
            self._ring.retain_from(overlap_start)
            # speech runs across the forced seam — reopen immediately, prepending
            # the overlap; no real pause here, so lead_pause is 0.
            self._seg_open = True
            self._seg_start_t = overlap_start
            self._seg_content_start_t = end_t
            self._cur_lead_pause = 0.0
            self._last_speech_t = end_t
        return event
