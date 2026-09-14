"""Shared segmenter event type.

Leaf module: imports nothing from the project. `segment/policy.py` (producer)
and `stt/scheduler.py` (consumer) both depend on it one-way, so no import cycle
can form between them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentClosed:
    """One closed segment of audio, ready to transcribe.

    Emitted by the VAD segmenter, consumed by the scheduler. Immutable because
    it flows through queues and an assembler concurrently.
    """

    seq: int  # monotonic, gap-free, session-relative
    audio: bytes  # int16 LE PCM, 16kHz mono (includes any prepended forced overlap)
    t_start: float  # session-relative seconds of the first sample in `audio`
    t_end: float  # session-relative seconds of the last sample
    lead_pause: float  # seconds of silence immediately before t_start (0.0 for seq 0)
    forced: bool  # True if closed by hitting MAX_SEGMENT, not a clean pause
