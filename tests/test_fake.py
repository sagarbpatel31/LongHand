"""FakeDictationClient: latency, out-of-order, 5xx, timeout, llm_error, retry."""

from __future__ import annotations

import asyncio

import pytest

from longhand.stt.client import DictationTimeout, ServerError, TranscriptionConfig
from longhand.stt.fake import FakeBehavior, FakeDictationClient, make_result


def _cfg(seq: int) -> TranscriptionConfig:
    return TranscriptionConfig(sample_rate=16000, channels=1, audio_format="pcm", seq=seq)


async def test_default_returns_a_transcript():
    c = FakeDictationClient()
    r = await c.transcribe(b"\x00\x01", _cfg(0))
    assert isinstance(r.text, str) and r.text
    assert c.calls[0].seq == 0
    assert c.calls[0].audio_len == 2


async def test_latency_produces_out_of_order_completion():
    c = FakeDictationClient(
        behaviors={
            0: FakeBehavior(result=make_result("first"), latency=0.05),
            1: FakeBehavior(result=make_result("second"), latency=0.01),
        }
    )
    completed: list[int] = []

    async def run(seq: int):
        await c.transcribe(b"\x00", _cfg(seq))
        completed.append(seq)

    await asyncio.gather(run(0), run(1))
    # seq 1 is faster, so it finishes first — responses out of submission order
    assert completed == [1, 0]


async def test_5xx_raises_server_error():
    c = FakeDictationClient(behaviors={0: FakeBehavior(raises=ServerError("boom"))})
    with pytest.raises(ServerError):
        await c.transcribe(b"\x00", _cfg(0))


async def test_timeout_raises():
    c = FakeDictationClient(behaviors={0: FakeBehavior(raises=DictationTimeout("slow"))})
    with pytest.raises(DictationTimeout):
        await c.transcribe(b"\x00", _cfg(0))


async def test_llm_error_returned_as_success():
    c = FakeDictationClient(
        behaviors={0: FakeBehavior(result=make_result("verbatim", llm_error="timeout"))}
    )
    r = await c.transcribe(b"\x00", _cfg(0))
    assert r.llm_error == "timeout"
    assert r.best_text == "verbatim"


async def test_transient_then_success_for_retry():
    c = FakeDictationClient(
        behaviors={
            0: FakeBehavior(
                transient=[ServerError(), ServerError()],
                result=make_result("recovered"),
            )
        }
    )
    for _ in range(2):
        with pytest.raises(ServerError):
            await c.transcribe(b"\x00", _cfg(0))
    r = await c.transcribe(b"\x00", _cfg(0))
    assert r.text == "recovered"
    assert len(c.calls) == 3
    assert [call.attempt for call in c.calls] == [0, 1, 2]


async def test_fake_enforces_config_contract():
    c = FakeDictationClient()
    bad = TranscriptionConfig(
        sample_rate=16000, channels=1, stt_prompt="a", prompt="b", seq=0
    )
    with pytest.raises(ValueError):
        await c.transcribe(b"\x00", bad)
