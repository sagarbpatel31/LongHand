"""FakeDictationClient — the offline stand-in used by every other test.

Implements the same `DictationClient` interface as the real client and can
simulate: latency, out-of-order returns (via per-seq latency), 5xx, timeouts,
`llm_error`, and transient-then-success (for retry tests). It touches no
network. Behaviours are keyed by `config.seq`; a default applies otherwise.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .client import (
    DictationClient,
    TranscriptionConfig,
    TranscriptionResult,
    Word,
)

logger = logging.getLogger(__name__)


def make_result(
    text: str,
    *,
    words: list[Word] | None = None,
    confidence: float = 1.0,
    llm_response: str | None = None,
    llm_error: str | None = None,
    audio_duration_ms: int | None = None,
    session_id: str = "fake-session",
    request_time_ms: int = 10,
    sync_time_ms: int = 10,
) -> TranscriptionResult:
    """Convenience builder for a canned result."""
    return TranscriptionResult(
        text=text,
        words=words if words is not None else [Word(t, 1.0) for t in text.split()],
        confidence=confidence,
        llm_response=llm_response,
        llm_error=llm_error,
        audio_duration_ms=audio_duration_ms,
        session_id=session_id,
        request_time_ms=request_time_ms,
        sync_time_ms=sync_time_ms,
        raw={},
    )


@dataclass
class FakeBehavior:
    """What the fake should do for a given seq.

    - `latency`: seconds to sleep before responding (drives out-of-order).
    - `raises`: an exception to raise on every attempt (permanent failure).
    - `transient`: exceptions raised on successive attempts, then `result`
      succeeds (drives retry tests).
    - `result`: what to return on success (defaults to a generic transcript).
    """

    result: TranscriptionResult | None = None
    latency: float = 0.0
    raises: Exception | None = None
    transient: list[Exception] = field(default_factory=list)


@dataclass
class FakeCall:
    seq: int | None
    audio_len: int
    attempt: int


class FakeDictationClient(DictationClient):
    def __init__(
        self,
        *,
        default: FakeBehavior | None = None,
        behaviors: dict[int, FakeBehavior] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.default = default or FakeBehavior(result=make_result("fake transcript"))
        self.behaviors = behaviors or {}
        self.calls: list[FakeCall] = []
        self._attempts: dict[int | None, int] = {}
        self._sleep = sleep

    async def transcribe(
        self, audio: bytes, config: TranscriptionConfig
    ) -> TranscriptionResult:
        # Same contract as the real client: config invariants are enforced.
        config.validate()

        seq = config.seq
        attempt = self._attempts.get(seq, 0)
        self._attempts[seq] = attempt + 1
        self.calls.append(FakeCall(seq=seq, audio_len=len(audio), attempt=attempt))

        beh = self.behaviors.get(seq, self.default)
        if beh.latency:
            await self._sleep(beh.latency)

        logger.debug("seq=%s fake.transcribe attempt=%d", seq, attempt)

        if attempt < len(beh.transient):
            raise beh.transient[attempt]
        if beh.raises is not None:
            raise beh.raises
        return beh.result if beh.result is not None else make_result("fake transcript")
