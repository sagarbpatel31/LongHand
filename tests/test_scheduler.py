"""SegmentScheduler: ordering (§9.4), failure (§9.6), retry (§9.7),
backpressure (§9.8), concurrency cap, and cancellation cleanup (§12).

All offline — FakeDictationClient only, injected sleep for deterministic backoff.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from longhand.assemble.glossary import Glossary
from longhand.assemble.order import OrderedAssembler
from longhand.segment.types import SegmentClosed
from longhand.stt.client import (
    BadRequestError,
    DictationClient,
    ServerError,
    TranscriptionConfig,
)
from longhand.stt.fake import FakeBehavior, FakeDictationClient, make_result
from longhand.stt.scheduler import SegmentScheduler, transcribe_segments

BASE_CFG = TranscriptionConfig(sample_rate=16000, channels=1, audio_format="pcm")


def _seg(seq: int, *, lead_pause: float = 0.0, forced: bool = False, audio: bytes = b"aud") -> SegmentClosed:
    return SegmentClosed(
        seq=seq,
        audio=audio,
        t_start=float(seq),
        t_end=float(seq) + 1.0,
        lead_pause=lead_pause,
        forced=forced,
    )


async def _noop_sleep(_d: float) -> None:
    return None


# --- happy path -------------------------------------------------------------


async def test_all_segments_transcribed_in_order():
    fake = FakeDictationClient(
        behaviors={i: FakeBehavior(result=make_result(f"seg{i}")) for i in range(5)}
    )
    doc = await transcribe_segments(
        [_seg(i) for i in range(5)], fake, base_config=BASE_CFG, sleep=_noop_sleep
    )
    assert [s.seq for s in doc] == [0, 1, 2, 3, 4]
    assert [s.best_text for s in doc] == [f"seg{i}" for i in range(5)]
    assert all(not s.is_gap for s in doc)


# --- retry (§9.7) -----------------------------------------------------------


async def test_retry_on_retryable_then_success():
    sleeps: list[float] = []

    async def rec_sleep(d: float) -> None:
        sleeps.append(d)

    fake = FakeDictationClient(
        behaviors={
            2: FakeBehavior(
                transient=[ServerError(), ServerError()],
                result=make_result("recovered"),
            )
        }
    )
    doc = await transcribe_segments(
        [_seg(i) for i in range(4)], fake, base_config=BASE_CFG, sleep=rec_sleep
    )
    # 3 attempts on seq 2, all re-sent the same retained bytes
    seq2_calls = [c for c in fake.calls if c.seq == 2]
    assert len(seq2_calls) == 3
    assert {c.audio_len for c in seq2_calls} == {len(b"aud")}
    assert sleeps == [0.5, 1.0]  # exponential backoff schedule
    assert doc[2].best_text == "recovered"
    assert not doc[2].is_gap


async def test_non_retryable_not_retried_becomes_gap():
    fake = FakeDictationClient(behaviors={0: FakeBehavior(raises=BadRequestError("bad"))})
    doc = await transcribe_segments([_seg(0)], fake, base_config=BASE_CFG, sleep=_noop_sleep)
    assert len([c for c in fake.calls if c.seq == 0]) == 1  # no retries
    assert doc[0].is_gap
    assert doc[0].error == "BadRequestError"


# --- permanent failure survives (§9.6) --------------------------------------


async def test_permanent_failure_becomes_gap_document_survives():
    fake = FakeDictationClient(
        default=FakeBehavior(result=make_result("ok")),
        behaviors={2: FakeBehavior(raises=ServerError("down"))},
    )
    doc = await transcribe_segments(
        [_seg(i) for i in range(5)], fake, base_config=BASE_CFG, sleep=_noop_sleep
    )
    assert [s.seq for s in doc] == [0, 1, 2, 3, 4]
    assert doc[2].is_gap and doc[2].best_text == ""
    assert len([c for c in fake.calls if c.seq == 2]) == 3  # retried to exhaustion
    # neighbors intact
    assert all(doc[i].best_text == "ok" for i in (0, 1, 3, 4))


# --- concurrency cap --------------------------------------------------------


async def test_never_exceeds_four_in_flight():
    in_flight = 0
    peak = 0

    async def gated_sleep(_d: float) -> None:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)  # let all admitted workers overlap
        in_flight -= 1

    # latency triggers the fake's injected sleep while holding the cap slot
    fake = FakeDictationClient(default=FakeBehavior(latency=0.001), sleep=gated_sleep)
    asm = OrderedAssembler()
    sched = SegmentScheduler(fake, base_config=BASE_CFG, assembler=asm)
    for i in range(20):
        await sched.submit(_seg(i))
    await sched.close()
    await sched.run()

    assert peak == 4  # cap saturated but never exceeded (fake-side observation)
    assert sched.peak_in_flight == 4  # scheduler-side observation
    assert [s.seq for s in asm.document()] == list(range(20))


# --- backpressure (§9.8) ----------------------------------------------------


async def test_backpressure_signals_when_queue_exceeds_eight():
    depths: list[int] = []
    release = asyncio.Event()

    async def gated_sleep(_d: float) -> None:
        await release.wait()

    # every request blocks until released -> queue piles up behind the cap
    fake = FakeDictationClient(default=FakeBehavior(latency=1.0), sleep=gated_sleep)
    asm = OrderedAssembler()
    sched = SegmentScheduler(
        fake, base_config=BASE_CFG, assembler=asm, on_backpressure=depths.append
    )
    runner = asyncio.create_task(sched.run())
    for i in range(20):
        await sched.submit(_seg(i))

    assert depths, "backpressure callback never fired"
    assert max(depths) > 8

    release.set()
    await sched.close()
    await runner
    assert [s.seq for s in asm.emit_all()] == list(range(20))


# --- reverse completion still ordered (§9.4) --------------------------------


async def test_reverse_completion_still_ordered():
    # seq 0 slowest ... seq 3 fastest -> completions scrambled, output ordered
    behaviors = {
        i: FakeBehavior(result=make_result(f"seg{i}"), latency=(4 - i) * 0.01)
        for i in range(4)
    }
    fake = FakeDictationClient(behaviors=behaviors)  # real asyncio.sleep latency
    doc = await transcribe_segments([_seg(i) for i in range(4)], fake, base_config=BASE_CFG)
    assert [s.seq for s in doc] == [0, 1, 2, 3]
    assert [s.best_text for s in doc] == [f"seg{i}" for i in range(4)]


# --- cancellation cleanup (§12) ---------------------------------------------


class _SpyClient(DictationClient):
    def __init__(self) -> None:
        self.closed = False
        self.started = 0

    async def transcribe(self, audio, config):
        self.started += 1
        await asyncio.sleep(10)  # long — will be cancelled
        return make_result("never")

    async def aclose(self) -> None:
        self.closed = True


async def test_aclose_cancels_in_flight_and_closes_client():
    spy = _SpyClient()
    sched = SegmentScheduler(spy, base_config=BASE_CFG)
    await sched.submit(_seg(0))
    await asyncio.sleep(0.01)  # let the worker enter transcribe
    assert spy.started == 1

    await sched.aclose()
    assert spy.closed is True
    assert not sched._tasks  # no leaked tasks


# --- carryover hooks (§5) ---------------------------------------------------


class _RecordingClient(DictationClient):
    def __init__(self) -> None:
        self.configs: list[TranscriptionConfig] = []

    async def transcribe(self, audio, config):
        self.configs.append(config)
        return make_result(f"seg{config.seq}")


async def test_default_prepare_config_unchanged():
    c = _RecordingClient()
    await transcribe_segments([_seg(i) for i in range(3)], c, base_config=BASE_CFG)
    assert sorted(cfg.seq for cfg in c.configs) == [0, 1, 2]
    assert all(cfg.stt_prompt is None and cfg.keyterms_prompt is None for cfg in c.configs)


async def test_prepare_config_invoked_per_segment():
    c = _RecordingClient()

    def prep(seg: SegmentClosed) -> TranscriptionConfig:
        return replace(BASE_CFG, seq=seg.seq, stt_prompt=f"ctx{seg.seq}")

    await transcribe_segments([_seg(i) for i in range(3)], c, base_config=BASE_CFG, prepare_config=prep)
    by_seq = {cfg.seq: cfg for cfg in c.configs}
    assert by_seq[1].stt_prompt == "ctx1"


async def test_prepare_config_seq_is_forced():
    c = _RecordingClient()
    doc = await transcribe_segments(
        [_seg(0), _seg(1)],
        c,
        base_config=BASE_CFG,
        prepare_config=lambda seg: replace(BASE_CFG, seq=999),  # wrong seq
    )
    assert [s.seq for s in doc] == [0, 1]  # assembler used seg.seq
    assert sorted(cfg.seq for cfg in c.configs) == [0, 1]  # scheduler forced it back


async def test_on_result_fires_once_per_success_not_gap():
    seen: list[int] = []
    fake = FakeDictationClient(
        default=FakeBehavior(result=make_result("ok")),
        behaviors={1: FakeBehavior(raises=ServerError("down"))},
    )
    await transcribe_segments(
        [_seg(i) for i in range(3)],
        fake,
        base_config=BASE_CFG,
        sleep=_noop_sleep,
        on_result=lambda seq, r: seen.append(seq),
    )
    assert sorted(seen) == [0, 2]  # seq 1 is a gap -> no on_result


class _TermClient(DictationClient):
    def __init__(self) -> None:
        self.configs: list[TranscriptionConfig] = []

    async def transcribe(self, audio, config):
        self.configs.append(config)
        text = "Kubernetes rocks" if config.seq == 0 else "more text here"
        return make_result(text)


async def test_glossary_carryover_later_segment_sees_earlier_terms():
    g = Glossary()
    c = _TermClient()

    def prep(seg: SegmentClosed) -> TranscriptionConfig:
        return replace(BASE_CFG, seq=seg.seq, keyterms_prompt=g.keyterms() or None)

    await transcribe_segments(
        [_seg(0), _seg(1)],
        c,
        base_config=BASE_CFG,
        prepare_config=prep,
        on_result=lambda seq, r: g.observe(r.best_text),
        max_in_flight=1,  # serial -> deterministic causal carryover
    )
    cfg1 = next(cfg for cfg in c.configs if cfg.seq == 1)
    assert cfg1.keyterms_prompt is not None
    assert any(t.lower() == "kubernetes" for t in cfg1.keyterms_prompt)
