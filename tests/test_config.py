"""Config invariants: alias exclusivity, caps, client-side field stripping."""

from __future__ import annotations

import pytest

from longhand.stt.client import TranscriptionConfig


def _base(**kw) -> TranscriptionConfig:
    return TranscriptionConfig(sample_rate=16000, channels=1, **kw)


def test_valid_minimal_config():
    _base().validate()  # does not raise


def test_stt_prompt_and_prompt_mutually_exclusive():
    with pytest.raises(ValueError):
        _base(stt_prompt="a", prompt="b").validate()


def test_keyterms_aliases_mutually_exclusive():
    with pytest.raises(ValueError):
        _base(keyterms=["x"], word_boost=["y"]).validate()
    with pytest.raises(ValueError):
        _base(keyterms_prompt=["x"], keyterms=["y"]).validate()


def test_keyterms_term_cap():
    # 101 short terms: trips the 100-term cap, not the char cap
    with pytest.raises(ValueError):
        _base(keyterms_prompt=[f"t{i}" for i in range(101)]).validate()


def test_keyterms_char_cap():
    # one very long term: trips the 8000-char cap, not the term cap
    with pytest.raises(ValueError):
        _base(keyterms_prompt=["x" * 8001]).validate()


def test_keyterms_at_caps_ok():
    _base(keyterms_prompt=[f"{i:03d}" for i in range(100)]).validate()


def test_stt_prompt_char_cap():
    with pytest.raises(ValueError):
        _base(stt_prompt="x" * 6001).validate()


def test_llm_instruction_char_cap():
    with pytest.raises(ValueError):
        _base(llm_instruction="x" * 2049).validate()


def test_sample_rate_and_channels_required():
    with pytest.raises(ValueError):
        TranscriptionConfig(sample_rate=0, channels=1).validate()
    with pytest.raises(ValueError):
        TranscriptionConfig(sample_rate=16000, channels=0).validate()


def test_audio_format_must_be_pcm_or_wav():
    with pytest.raises(ValueError):
        _base(audio_format="mp3").validate()


def test_to_json_dict_excludes_client_side_fields():
    cfg = _base(audio_format="pcm", seq=7, stt_prompt="hi", language_codes=["en"])
    d = cfg.to_json_dict()
    assert "audio_format" not in d
    assert "seq" not in d
    assert d == {
        "sample_rate": 16000,
        "channels": 1,
        "language_codes": ["en"],
        "stt_prompt": "hi",
    }
