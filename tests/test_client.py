"""Client guards + response mapping. No network: httpx.MockTransport only."""

from __future__ import annotations

import io
import math
import struct
import wave

import httpx
import pytest

from longhand.stt.client import (
    AssemblyAIDictationClient,
    AuthError,
    BadRequestError,
    DictationTimeout,
    ServerError,
    TranscriptionConfig,
    TranscriptionResult,
    UnsupportedMediaError,
    Word,
    audio_duration_seconds,
    build_multipart,
    raise_for_status,
)


def _make_wav(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    n = int(seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = b"".join(
            struct.pack("<h", int(0.1 * 32767 * math.sin(2 * math.pi * 220 * i / sample_rate)))
            for i in range(n)
        )
        w.writeframes(frames)
    return buf.getvalue()


def _client(handler) -> AssemblyAIDictationClient:
    return AssemblyAIDictationClient(
        api_key="RAWKEY",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _pcm_cfg(seq: int | None = None) -> TranscriptionConfig:
    return TranscriptionConfig(sample_rate=16000, channels=1, audio_format="pcm", seq=seq)


# --- multipart part ordering (load-bearing) ---------------------------------


def test_multipart_config_precedes_audio():
    body, ct = build_multipart(
        {"sample_rate": 16000, "channels": 1},
        b"\x00\x01\x02\x03",
        audio_filename="audio.pcm",
        audio_content_type="application/octet-stream",
        boundary="BOUND",
    )
    assert b'name="config"' in body
    assert b'name="audio"' in body
    assert body.index(b'name="config"') < body.index(b'name="audio"')
    assert ct == "multipart/form-data; boundary=BOUND"
    # two part headers + one closing delimiter
    assert body.count(b"--BOUND") == 3


# --- duration helpers -------------------------------------------------------


def test_pcm_duration():
    # 16kHz mono 16-bit → 1s == 32000 bytes
    assert abs(audio_duration_seconds(bytes(32000), _pcm_cfg()) - 1.0) < 1e-9


def test_wav_duration():
    audio = _make_wav(seconds=0.5)
    cfg = TranscriptionConfig(sample_rate=16000, channels=1, audio_format="wav")
    assert abs(audio_duration_seconds(audio, cfg) - 0.5) < 1e-3


# --- 110s ceiling fires BEFORE any network ----------------------------------


async def test_ceiling_guard_blocks_before_network():
    def handler(request):  # pragma: no cover - must never run
        raise AssertionError("network was touched despite over-length audio")

    client = _client(handler)
    # 111s of PCM > 110s ceiling
    audio = bytes(16000 * 2 * 111)
    with pytest.raises(ValueError):
        await client.transcribe(audio, _pcm_cfg(seq=0))


# --- happy path: raw auth header, part ordering on the wire, mapping --------


async def test_transcribe_sends_raw_auth_and_maps_response():
    captured: dict = {}

    def handler(request):
        captured["auth"] = request.headers.get("authorization")
        captured["ctype"] = request.headers.get("content-type")
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "text": "hello world",
                "words": [{"text": "hello", "confidence": 0.9}],
                "confidence": 0.95,
                "llm_response": "Hello world.",
                "llm_error": None,
                "audio_duration_ms": 1000,
                "session_id": "s1",
                "request_time_ms": 40,
                "sync_time_ms": 55,
            },
        )

    client = _client(handler)
    res = await client.transcribe(bytes(32000), _pcm_cfg(seq=3))

    assert isinstance(res, TranscriptionResult)
    assert res.text == "hello world"
    assert res.best_text == "Hello world."
    assert res.words == [Word("hello", 0.9)]
    assert res.audio_duration_ms == 1000
    # RAW key — no Bearer prefix
    assert captured["auth"] == "RAWKEY"
    assert not captured["auth"].lower().startswith("bearer")
    assert captured["ctype"].startswith("multipart/form-data")
    # config still precedes audio on the wire
    assert captured["body"].index(b'name="config"') < captured["body"].index(b'name="audio"')


# --- status mapping ---------------------------------------------------------


@pytest.mark.parametrize(
    "status,exc",
    [
        (404, AuthError),  # bad key returns 404, not 401
        (400, BadRequestError),
        (415, UnsupportedMediaError),
        (500, ServerError),
        (503, ServerError),
    ],
)
async def test_status_maps_to_exception(status, exc):
    client = _client(lambda request: httpx.Response(status, text="err"))
    with pytest.raises(exc):
        await client.transcribe(bytes(32000), _pcm_cfg(seq=0))


def test_raise_for_status_200_is_clean():
    assert raise_for_status(200) is None


def test_retryable_classification():
    assert ServerError().retryable is True
    assert DictationTimeout().retryable is True
    assert AuthError().retryable is False
    assert BadRequestError().retryable is False


# --- llm_error is a success, not a failure ----------------------------------


async def test_llm_error_is_success_and_falls_back_to_text():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "text": "verbatim text",
                "words": [],
                "confidence": 0.9,
                "llm_response": None,
                "llm_error": "timeout",
                "audio_duration_ms": 900,
            },
        )

    client = _client(handler)
    res = await client.transcribe(bytes(32000), _pcm_cfg(seq=1))
    assert res.llm_error == "timeout"  # no exception raised
    assert res.best_text == "verbatim text"  # fell back to verbatim


# --- transport timeout maps to a retryable DictationTimeout -----------------


async def test_timeout_maps_to_dictation_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    client = _client(handler)
    with pytest.raises(DictationTimeout):
        await client.transcribe(bytes(32000), _pcm_cfg(seq=2))
