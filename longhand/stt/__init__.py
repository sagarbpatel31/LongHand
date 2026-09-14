"""Speech-to-text: the Dictation client interface, real impl, and fake."""

from .client import (
    AssemblyAIDictationClient,
    AuthError,
    BadRequestError,
    DictationClient,
    DictationError,
    DictationTimeout,
    ServerError,
    TranscriptionConfig,
    TranscriptionResult,
    UnsupportedMediaError,
    Word,
)
from .fake import FakeBehavior, FakeDictationClient, make_result

__all__ = [
    "AssemblyAIDictationClient",
    "AuthError",
    "BadRequestError",
    "DictationClient",
    "DictationError",
    "DictationTimeout",
    "FakeBehavior",
    "FakeDictationClient",
    "ServerError",
    "TranscriptionConfig",
    "TranscriptionResult",
    "UnsupportedMediaError",
    "Word",
    "make_result",
]
