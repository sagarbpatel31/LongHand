"""Rolling audio buffer — retains audio until a segment is confirmed.

The single owner of segment audio bytes and the single place the memory bound is
enforced. Audio is pinned only while it is still needed:

- everything before ``confirm(t)`` is eligible to drop, and
- ``retain_from(t)`` pins the forced-cut overlap tail so it survives a confirm.

Because the scheduler confirms a segment right after copying its bytes into a
`SegmentClosed`, the resident audio stays bounded (§9.9): 20 min @ 16kHz mono
int16 would be ~38MB if never pruned; with prompt confirms it holds only the
unconfirmed tail plus any pinned overlap.
"""

from __future__ import annotations

_BYTES_PER_SAMPLE = 2  # 16-bit PCM


class RingBuffer:
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        max_seconds: float = 30.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_bytes = channels * _BYTES_PER_SAMPLE
        self._max_seconds = max_seconds  # advisory window hint
        self._buf = bytearray()
        self._base_sample = 0  # absolute index of the first resident sample
        self._confirmed_upto = 0  # absolute sample; everything before is droppable
        self._retain_watermark: int | None = None  # pinned-from sample, or None

    # --- geometry -----------------------------------------------------------

    def _abs_sample(self, t: float) -> int:
        return round(t * self._sample_rate)

    @property
    def _write_sample(self) -> int:
        return self._base_sample + len(self._buf) // self._frame_bytes

    @property
    def nbytes(self) -> int:
        return len(self._buf)

    @property
    def duration_s(self) -> float:
        return (len(self._buf) // self._frame_bytes) / self._sample_rate

    # --- writing ------------------------------------------------------------

    def write(self, frame: bytes | bytearray | "object") -> None:
        """Append captured int16 audio (bytes or a numpy int16 array)."""
        if isinstance(frame, (bytes, bytearray)):
            data = bytes(frame)
        else:  # numpy array or anything with tobytes()
            data = frame.tobytes()  # type: ignore[attr-defined]
        self._buf.extend(data)
        self._prune()

    # --- lifecycle ----------------------------------------------------------

    def confirm(self, t: float) -> None:
        """Mark audio before session-time `t` as no longer needed."""
        self._confirmed_upto = max(self._confirmed_upto, self._abs_sample(t))
        self._prune()

    def retain_from(self, t: float) -> None:
        """Pin audio from session-time `t` onward (forced-overlap tail)."""
        self._retain_watermark = self._abs_sample(t)
        self._prune()

    def _prune(self) -> None:
        keep_from = self._confirmed_upto
        if self._retain_watermark is not None:
            keep_from = min(keep_from, self._retain_watermark)
        keep_from = min(keep_from, self._write_sample)
        drop = keep_from - self._base_sample
        if drop > 0:
            del self._buf[: drop * self._frame_bytes]
            self._base_sample = keep_from

    # --- reading ------------------------------------------------------------

    def slice(self, t_start: float, t_end: float) -> bytes:
        """Return int16 LE PCM bytes for [t_start, t_end). Raises if pruned."""
        s0 = self._abs_sample(t_start)
        s1 = self._abs_sample(t_end)
        if s0 < self._base_sample:
            raise ValueError(
                f"audio before t={t_start:.3f}s was already pruned "
                f"(base={self._base_sample / self._sample_rate:.3f}s)"
            )
        if s1 > self._write_sample:
            raise ValueError(
                f"audio through t={t_end:.3f}s not yet written "
                f"(write={self._write_sample / self._sample_rate:.3f}s)"
            )
        off0 = (s0 - self._base_sample) * self._frame_bytes
        off1 = (s1 - self._base_sample) * self._frame_bytes
        return bytes(self._buf[off0:off1])
