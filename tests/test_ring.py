"""RingBuffer: slice correctness, retention, and the memory bound (§9.9)."""

from __future__ import annotations

import pytest

from longhand.capture.ring import RingBuffer

RATE = 16000
FRAME_BYTES = 2  # mono int16


def _pcm(n_samples: int, fill: int = 0) -> bytes:
    return bytes([fill & 0xFF, (fill >> 8) & 0xFF]) * n_samples


def test_slice_returns_expected_byte_length():
    ring = RingBuffer(sample_rate=RATE)
    ring.write(_pcm(RATE))  # 1.0s
    out = ring.slice(0.0, 0.5)
    assert len(out) == int(0.5 * RATE) * FRAME_BYTES


def test_slice_matches_written_content():
    ring = RingBuffer(sample_rate=RATE)
    # distinct byte pattern per 0.1s block so a slice is identifiable
    blocks = [_pcm(RATE // 10, fill=i) for i in range(5)]  # 0.5s total
    for b in blocks:
        ring.write(b)
    assert ring.slice(0.2, 0.3) == blocks[2]


def test_prune_after_confirm_frees_memory():
    ring = RingBuffer(sample_rate=RATE)
    ring.write(_pcm(RATE))  # 1.0s
    before = ring.nbytes
    ring.confirm(0.8)  # release first 0.8s
    assert ring.nbytes < before
    assert abs(ring.duration_s - 0.2) < 1e-3


def test_forced_overlap_tail_retained_across_confirm():
    ring = RingBuffer(sample_rate=RATE)
    ring.write(_pcm(RATE))  # 1.0s
    ring.retain_from(0.5)  # pin the last 0.5s
    ring.confirm(0.9)  # would otherwise drop through 0.9s
    # retained tail still sliceable
    assert len(ring.slice(0.5, 1.0)) == int(0.5 * RATE) * FRAME_BYTES


def test_slice_after_prune_raises():
    ring = RingBuffer(sample_rate=RATE)
    ring.write(_pcm(RATE))
    ring.confirm(0.5)
    with pytest.raises(ValueError):
        ring.slice(0.1, 0.4)  # pruned


def test_slice_beyond_written_raises():
    ring = RingBuffer(sample_rate=RATE)
    ring.write(_pcm(RATE // 2))  # 0.5s
    with pytest.raises(ValueError):
        ring.slice(0.4, 0.9)


def test_memory_bounded_over_long_run():
    # §9.9 — 20 min of writes with prompt confirms stays bounded (would be ~38MB
    # unpruned). Write in 0.1s chunks; confirm keeping only the last ~0.5s.
    ring = RingBuffer(sample_rate=RATE)
    chunk_samples = RATE // 10  # 0.1s
    total_chunks = 20 * 60 * 10  # 20 minutes
    keep_s = 0.5
    cap_bytes = int(2.0 * RATE) * FRAME_BYTES  # generous 2s ceiling
    peak = 0
    for i in range(total_chunks):
        ring.write(_pcm(chunk_samples, fill=i))
        t_now = (i + 1) * 0.1
        ring.confirm(max(0.0, t_now - keep_s))
        peak = max(peak, ring.nbytes)
    assert peak <= cap_bytes  # never grows unbounded
    assert ring.duration_s <= keep_s + 0.15
