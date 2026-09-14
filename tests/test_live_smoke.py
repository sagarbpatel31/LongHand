"""The ONE live smoke test. Excluded by default; run with `pytest -m live`.

Sends a short synthetic WAV to the real Dictation API and asserts the response
SHAPE (not exact text — a tone has no words). Skips if no key is present.
"""

from __future__ import annotations

import io
import math
import struct
import wave

import pytest

from longhand.settings import get_api_key
from longhand.stt.client import AssemblyAIDictationClient, TranscriptionConfig, TranscriptionResult

pytestmark = pytest.mark.live


def _make_wav(seconds: float = 2.0, sample_rate: int = 16000, freq: float = 220.0) -> bytes:
    n = int(seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = b"".join(
            struct.pack("<h", int(0.2 * 32767 * math.sin(2 * math.pi * freq * i / sample_rate)))
            for i in range(n)
        )
        w.writeframes(frames)
    return buf.getvalue()


async def test_live_smoke_short_wav():
    key = get_api_key()
    if not key:
        pytest.skip("no ASSEMBLYAI_API_KEY in env")

    audio = _make_wav(seconds=2.0)
    cfg = TranscriptionConfig(sample_rate=16000, channels=1, audio_format="wav")
    client = AssemblyAIDictationClient(api_key=key)
    try:
        await client.warm()
        res = await client.transcribe(audio, cfg)
    finally:
        await client.aclose()

    assert isinstance(res, TranscriptionResult)
    assert isinstance(res.text, str)
    assert isinstance(res.words, list)
    assert res.audio_duration_ms is not None
    # llm_error is allowed to be set — that is still a successful request.
