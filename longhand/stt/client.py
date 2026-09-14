"""Async Dictation client — interface, real implementation, guards.

Every "hard API fact" from CLAUDE.md is encoded here as either a guard that
fires before the network is touched, or a mapping applied to the response.
The interface (`DictationClient`) exists so `FakeDictationClient` can stand in
for the real thing in every offline test. See stt/fake.py.
"""

from __future__ import annotations

import io
import logging
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx

from ..settings import get_api_key

logger = logging.getLogger(__name__)

# --- Hard API facts (do not re-derive) --------------------------------------

DICTATION_URL = "https://dictation.assemblyai.com/v1/transcribe/live"
HTTP_TIMEOUT_S = 90.0
#: 120s API cap, minus a 10s safety margin. Never send more than this.
MAX_SEGMENT_SECONDS = 110.0
_BYTES_PER_SAMPLE = 2  # 16-bit PCM
_STT_PROMPT_MAX_CHARS = 6000
_LLM_INSTRUCTION_MAX_CHARS = 2048
_KEYTERMS_MAX_TERMS = 100
_KEYTERMS_MAX_CHARS = 8000


# --- Exceptions -------------------------------------------------------------


class DictationError(Exception):
    """Base for all Dictation client errors."""

    retryable: bool = False


class AuthError(DictationError):
    """Invalid / missing key. NOTE: the API returns 404 (not 401) for this."""


class BadRequestError(DictationError):
    """400 — usually part ordering, missing config, or conflicting aliases."""


class UnsupportedMediaError(DictationError):
    """415 — audio was not WAV or raw 16-bit PCM."""


class ServerError(DictationError):
    """5xx — transient, safe to retry from retained audio."""

    retryable = True


class DictationTimeout(DictationError):
    """HTTP timeout — transient, safe to retry from retained audio."""

    retryable = True


# --- Data model -------------------------------------------------------------


@dataclass
class Word:
    text: str
    confidence: float


@dataclass
class TranscriptionResult:
    text: str  # verbatim, never LLM-touched
    words: list[Word]
    confidence: float | None
    llm_response: str | None
    llm_error: str | None
    audio_duration_ms: int | None
    session_id: str | None
    request_time_ms: int | None
    sync_time_ms: int | None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "TranscriptionResult":
        words = [
            Word(text=w.get("text", ""), confidence=w.get("confidence", 0.0))
            for w in (d.get("words") or [])
        ]
        return cls(
            text=d.get("text", "") or "",
            words=words,
            confidence=d.get("confidence"),
            llm_response=d.get("llm_response"),
            llm_error=d.get("llm_error"),
            audio_duration_ms=d.get("audio_duration_ms"),
            session_id=d.get("session_id"),
            request_time_ms=d.get("request_time_ms"),
            sync_time_ms=d.get("sync_time_ms"),
            raw=d,
        )

    @property
    def best_text(self) -> str:
        """Cleaned rewrite if present, else the verbatim transcript.

        The rewrite is best-effort: a non-null `llm_error` with a null
        `llm_response` is a *success*, not a failure — we simply fall back.
        """
        return self.llm_response if self.llm_response else self.text


@dataclass
class TranscriptionConfig:
    """Config for one segment.

    `audio_format` and `seq` are client-side only and are NOT serialized into
    the request body — `seq` correlates logs/fakes, `audio_format` selects the
    part content-type and drives the duration guard.
    """

    sample_rate: int
    channels: int = 1
    audio_format: str = "pcm"  # "pcm" (raw 16-bit LE) | "wav"
    language_codes: list[str] | None = None
    stt_prompt: str | None = None
    prompt: str | None = None  # alias of stt_prompt — send only one
    keyterms_prompt: list[str] | None = None
    keyterms: list[str] | None = None  # alias — send only one of the three
    word_boost: list[str] | None = None  # alias — send only one of the three
    llm_instruction: str | None = None
    seq: int | None = None  # client-side correlation id

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate is required and must be > 0")
        if self.channels <= 0:
            raise ValueError("channels is required and must be > 0")
        if self.audio_format not in ("pcm", "wav"):
            raise ValueError("audio_format must be 'pcm' or 'wav' (WAV/PCM only)")

        # Mutually exclusive prompt aliases.
        if self.stt_prompt is not None and self.prompt is not None:
            raise ValueError("send only ONE of stt_prompt / prompt")
        for name, val, cap in (
            ("stt_prompt", self.stt_prompt, _STT_PROMPT_MAX_CHARS),
            ("prompt", self.prompt, _STT_PROMPT_MAX_CHARS),
            ("llm_instruction", self.llm_instruction, _LLM_INSTRUCTION_MAX_CHARS),
        ):
            if val is not None and len(val) > cap:
                raise ValueError(f"{name} exceeds {cap} chars")

        # Mutually exclusive keyterm aliases.
        aliases = [
            a
            for a in (self.keyterms_prompt, self.keyterms, self.word_boost)
            if a is not None
        ]
        if len(aliases) > 1:
            raise ValueError(
                "send only ONE of keyterms_prompt / keyterms / word_boost"
            )
        if aliases:
            terms = aliases[0]
            if len(terms) > _KEYTERMS_MAX_TERMS:
                raise ValueError(f"keyterms cap: at most {_KEYTERMS_MAX_TERMS} terms")
            if sum(len(t) for t in terms) > _KEYTERMS_MAX_CHARS:
                raise ValueError(f"keyterms cap: at most {_KEYTERMS_MAX_CHARS} chars")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize ONLY real API fields — client-side fields are dropped."""
        d: dict[str, Any] = {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }
        if self.language_codes:
            d["language_codes"] = self.language_codes
        if self.stt_prompt is not None:
            d["stt_prompt"] = self.stt_prompt
        if self.prompt is not None:
            d["prompt"] = self.prompt
        if self.keyterms_prompt is not None:
            d["keyterms_prompt"] = self.keyterms_prompt
        if self.keyterms is not None:
            d["keyterms"] = self.keyterms
        if self.word_boost is not None:
            d["word_boost"] = self.word_boost
        if self.llm_instruction is not None:
            d["llm_instruction"] = self.llm_instruction
        return d


# --- Pure helpers (unit-testable without a network) -------------------------


def audio_duration_seconds(audio: bytes, config: TranscriptionConfig) -> float:
    """Duration of `audio` given the config's format.

    WAV: parsed from the header. PCM: derived from length, rate, channels,
    and 16-bit sample width.
    """
    if config.audio_format == "wav":
        with wave.open(io.BytesIO(audio), "rb") as w:
            rate = w.getframerate() or config.sample_rate
            return w.getnframes() / float(rate)
    denom = config.sample_rate * config.channels * _BYTES_PER_SAMPLE
    return len(audio) / float(denom)


def build_multipart(
    config_json: dict[str, Any],
    audio: bytes,
    *,
    audio_filename: str,
    audio_content_type: str,
    boundary: str,
) -> tuple[bytes, str]:
    """Build a multipart/form-data body with `config` BEFORE `audio`.

    Part order is load-bearing: audio before config (or a missing config)
    yields a 400. We assemble the body by hand rather than relying on a dict
    to preserve order.
    """
    import json

    crlf = b"\r\n"
    b = boundary.encode("ascii")
    parts = [
        b"--" + b + crlf,
        b'Content-Disposition: form-data; name="config"' + crlf,
        b"Content-Type: application/json" + crlf,
        crlf,
        json.dumps(config_json).encode("utf-8") + crlf,
        b"--" + b + crlf,
        (
            f'Content-Disposition: form-data; name="audio"; '
            f'filename="{audio_filename}"'
        ).encode("utf-8")
        + crlf,
        f"Content-Type: {audio_content_type}".encode("utf-8") + crlf,
        crlf,
        audio + crlf,
        b"--" + b + b"--" + crlf,
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def raise_for_status(status_code: int, body: str = "") -> None:
    """Map a Dictation HTTP status to an exception. 200 returns cleanly."""
    if status_code == 200:
        return
    if status_code == 404:
        # The API returns 404 (not 401) for a bad/missing key.
        raise AuthError(f"auth failed (404): {body[:200]}")
    if status_code == 400:
        raise BadRequestError(f"bad request (400): {body[:200]}")
    if status_code == 415:
        raise UnsupportedMediaError(f"unsupported media (415): {body[:200]}")
    if 500 <= status_code < 600:
        raise ServerError(f"server error ({status_code}): {body[:200]}")
    raise DictationError(f"unexpected status {status_code}: {body[:200]}")


def _audio_part_meta(audio_format: str) -> tuple[str, str]:
    if audio_format == "wav":
        return "audio.wav", "audio/wav"
    # Raw 16-bit PCM MUST be `audio/pcm`. The API 415s `application/octet-stream`
    # ("cannot be decoded") — verified live.
    return "audio.pcm", "audio/pcm"


# --- Interface --------------------------------------------------------------


class DictationClient(ABC):
    """The interface every consumer codes against.

    Real and fake implementations are interchangeable; nothing downstream may
    depend on which one it has.
    """

    @abstractmethod
    async def transcribe(
        self, audio: bytes, config: TranscriptionConfig
    ) -> TranscriptionResult:
        """Transcribe one segment. Raises a DictationError subclass on failure."""

    async def warm(self) -> None:
        """Pay DNS/TCP/TLS up front. No-op by default."""

    async def aclose(self) -> None:
        """Release resources. No-op by default."""

    async def __aenter__(self) -> "DictationClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


# --- Real implementation ----------------------------------------------------


class AssemblyAIDictationClient(DictationClient):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DICTATION_URL,
        client: httpx.AsyncClient | None = None,
        timeout: float = HTTP_TIMEOUT_S,
    ) -> None:
        self._api_key = api_key or get_api_key()
        if not self._api_key:
            raise ValueError(
                "no AssemblyAI API key (pass api_key= or set ASSEMBLYAI_API_KEY)"
            )
        self._url = base_url
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def warm(self) -> None:
        try:
            await self._client.head(self._url)
        except httpx.HTTPError:
            # A non-2xx here (e.g. 404/405) is fine — the point is to open the
            # connection, not to succeed.
            pass

    async def transcribe(
        self, audio: bytes, config: TranscriptionConfig
    ) -> TranscriptionResult:
        config.validate()

        # Guard the 110s ceiling BEFORE touching the network.
        duration = audio_duration_seconds(audio, config)
        if duration > MAX_SEGMENT_SECONDS:
            raise ValueError(
                f"seq={config.seq} segment is {duration:.1f}s, exceeds the "
                f"{MAX_SEGMENT_SECONDS:.0f}s ceiling"
            )

        # Raw PCM requires sample_rate + channels — validate() already enforced
        # both are present and positive.
        filename, content_type = _audio_part_meta(config.audio_format)
        body, multipart_ct = build_multipart(
            config.to_json_dict(),
            audio,
            audio_filename=filename,
            audio_content_type=content_type,
            boundary=f"longhand-{uuid4().hex}",
        )
        headers = {
            "Authorization": self._api_key,  # RAW key — no "Bearer" prefix
            "Content-Type": multipart_ct,
        }

        logger.debug(
            "seq=%s transcribe.start dur=%.2fs bytes=%d fmt=%s",
            config.seq,
            duration,
            len(audio),
            config.audio_format,
        )
        try:
            resp = await self._client.post(self._url, content=body, headers=headers)
        except httpx.TimeoutException as e:
            logger.warning("seq=%s transcribe.timeout %s", config.seq, e)
            raise DictationTimeout(f"seq={config.seq} timed out: {e}") from e
        except httpx.HTTPError as e:
            logger.warning("seq=%s transcribe.http_error %s", config.seq, e)
            raise DictationError(f"seq={config.seq} transport error: {e}") from e

        raise_for_status(resp.status_code, resp.text)
        result = TranscriptionResult.from_json(resp.json())
        logger.debug(
            "seq=%s transcribe.ok llm_error=%s duration_ms=%s",
            config.seq,
            result.llm_error,
            result.audio_duration_ms,
        )
        return result

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
