"""Benchmark: WER metric, fixtures, fixture client, and A-vs-B (all offline)."""

from __future__ import annotations

from longhand.bench.fixture_client import FixtureDictationClient, words_lost
from longhand.bench.fixtures import build_fixture, demo_fixture
from longhand.bench.harness import compare, format_table
from longhand.bench.wer import edit_distance, wer
from longhand.segment.vad import FRAME_DURATION_S
from longhand.stt.client import TranscriptionConfig


# --- WER --------------------------------------------------------------------


def test_wer_identical_is_zero():
    assert wer("a b c", "a b c") == 0.0


def test_wer_one_substitution():
    assert abs(wer("a b c", "a x c") - 1 / 3) < 1e-9


def test_wer_empty_hyp_is_one():
    assert wer("a b c", "") == 1.0


def test_wer_normalizes_case_and_punct():
    assert wer("A, b! c.", "a b c") == 0.0


def test_edit_distance_insertion():
    assert edit_distance(["a", "b"], ["a", "b", "c"]) == 1


# --- fixtures ---------------------------------------------------------------


def test_fixture_ground_truth_and_timeline():
    fx = build_fixture(["a b c"], gaps=1.0)
    assert fx.ground_truth == "a b c"
    assert len(fx.words) == 3
    assert fx.words[0].t_start == 0.0
    assert abs(fx.words[0].t_end - 0.384) < 1e-9
    assert abs(fx.total_s - (3 * 0.384 + 1.0)) < 1e-9
    assert len(fx.audio) // 2 == round(fx.total_s * fx.sample_rate)
    assert len(fx.frame_script) == int(fx.total_s / FRAME_DURATION_S)
    assert any(fx.frame_script) and not all(fx.frame_script)  # speech + gap present


def test_demo_fixture_builds():
    fx = demo_fixture(n_clips=3, words_per_clip=18)
    assert fx.total_s > 0 and len(fx.words) == 3 * 18


# --- fixture client ---------------------------------------------------------


async def test_fixture_client_drops_straddling_word():
    fx = build_fixture(["a b c d e"], gaps=0.0)  # words at 0.384s each
    client = FixtureDictationClient(fx, {0: (0.0, 1.0)})
    cfg = TranscriptionConfig(sample_rate=16000, channels=1, audio_format="pcm", seq=0)
    res = await client.transcribe(b"", cfg)
    assert res.text == "a b c~"  # c straddles 1.0s -> mangled fragment


def test_words_lost_full_coverage_is_zero():
    fx = build_fixture(["a b c d e"], gaps=0.0)
    assert words_lost(fx, [(0.0, fx.total_s + 0.1)]) == 0


def test_words_lost_counts_straddler_once():
    fx = build_fixture(["a b c d e"], gaps=0.0)
    # split mid-word at 1.0 -> only "c" straddles; d,e covered by the 2nd range
    assert words_lost(fx, [(0.0, 1.0), (1.0, fx.total_s + 0.1)]) == 1


# --- A vs B -----------------------------------------------------------------


async def test_vad_beats_naive_chop():
    clip1 = " ".join(f"w{i}" for i in range(20))  # 20 words ~7.68s (> MIN_SEGMENT)
    clip2 = " ".join(f"w{i}" for i in range(20, 40))
    fx = build_fixture([clip1, clip2], gaps=1.0)  # 1.0s gap -> a clean VAD cut

    a, b = await compare(fx, chop_s=5.0)  # small chop forces mid-word straddles

    # B (cut at silence) loses nothing at seams; A (arbitrary chop) does.
    assert b.boundary_word_loss == 0
    assert a.boundary_word_loss > 0
    assert b.wer == 0.0
    assert b.wer < a.wer
    assert a.wer > 0.0
    # VAD actually segmented on the gap
    assert b.n_segments == 2


async def test_format_table_has_headline():
    fx = build_fixture([" ".join(f"w{i}" for i in range(18))], gaps=1.0)
    results = await compare(fx, chop_s=3.0)
    table = format_table(results, fx)
    assert "Headline" in table
    assert "WER" in table
