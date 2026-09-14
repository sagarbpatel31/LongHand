"""Voice-activity detection — interface + fake + Silero (onnxruntime) impl.

The `Vad` interface exists so every test uses `FakeVad` (scripted speech/silence)
and never touches onnxruntime, a model file, or a mic. `SileroVad` lazily imports
onnxruntime + the model inside `__init__`, so importing this module for a
FakeVad-only test costs nothing.

Frame contract: 512 samples == 32ms at 16kHz, float32 normalized to [-1, 1].
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

FRAME_SIZE = 512  # samples @ 16kHz == 32ms (Silero's required chunk size)
SAMPLE_RATE = 16000
FRAME_DURATION_S = FRAME_SIZE / SAMPLE_RATE  # 0.032


class Vad(ABC):
    frame_size: int = FRAME_SIZE
    sample_rate: int = SAMPLE_RATE

    @abstractmethod
    def is_speech(self, frame: np.ndarray) -> bool:
        """frame: float32 [-1,1], length == frame_size. Speech decision."""

    def speech_prob(self, frame: np.ndarray) -> float:
        """Richer 0..1 signal; default derives from is_speech."""
        return 1.0 if self.is_speech(frame) else 0.0

    def reset_states(self) -> None:
        """Clear recurrent state between sessions. No-op by default."""


def frames_from_spans(spans: Sequence[tuple[str, float]]) -> list[bool]:
    """Expand human-readable spans into a per-frame speech/silence script.

    e.g. [("speech", 8.0), ("silence", 0.7)] -> [True]*250 + [False]*22
    at 32ms/frame. The key test utility for both VAD and policy tests.
    """
    out: list[bool] = []
    for label, seconds in spans:
        if label not in ("speech", "silence"):
            raise ValueError(f"span label must be 'speech'|'silence', got {label!r}")
        n = round(seconds / FRAME_DURATION_S)
        out.extend([label == "speech"] * n)
    return out


class FakeVad(Vad):
    """Returns scripted decisions; ignores frame content. For tests only."""

    def __init__(self, script: list[bool] | Callable[[int], bool]) -> None:
        self._script = script
        self._i = 0

    def is_speech(self, frame: np.ndarray) -> bool:
        i = self._i
        self._i += 1
        if callable(self._script):
            return bool(self._script(i))
        if i >= len(self._script):
            return False  # past the end == silence
        return bool(self._script[i])

    def reset_states(self) -> None:
        self._i = 0


class SileroVad(Vad):
    """Silero VAD over onnxruntime (via silero-vad-notorch, no torch)."""

    def __init__(self, *, threshold: float = 0.5, model: object | None = None) -> None:
        self._threshold = threshold
        if model is None:
            # lazy: only paid when a real VAD is actually constructed
            from silero_vad_notorch import load_silero_vad

            model = load_silero_vad(onnx=True)
        self._model = model

    def speech_prob(self, frame: np.ndarray) -> float:
        x = np.ascontiguousarray(frame, dtype=np.float32)
        if x.shape[-1] != self.frame_size:
            raise ValueError(
                f"Silero needs {self.frame_size}-sample frames, got {x.shape[-1]}"
            )
        out = self._model(x, self.sample_rate)  # type: ignore[operator]
        return float(np.asarray(out).reshape(-1)[0])

    def is_speech(self, frame: np.ndarray) -> bool:
        return self.speech_prob(frame) >= self._threshold

    def reset_states(self) -> None:
        self._model.reset_states()  # type: ignore[attr-defined]
