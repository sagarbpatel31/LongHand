"""Benchmark: WER metric, fixtures, fixture client, and A-vs-B (all offline)."""

from __future__ import annotations

from longhand.bench.fixture_client import (
    DriftDictationClient,
    FixtureDictationClient,
    words_lost,
)
from longhand.bench.fixtures import DriftTerm, build_drift_fixture, build_fixture, demo_fixture
from longhand.bench.harness import compare, compare_drift, format_table
from longhand.bench.metrics import paragraph_boundary_f1, terminology_consistency_rate
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


# --- drift model + condition C ----------------------------------------------

_TERM = [DriftTerm("kubernetes", "koobernetes")]


def _one_clip_drift():
    # kubernetes at word-indices 0,2,4 -> ranks 0,1,2 (rank 2 drifts when unpinned)
    return build_drift_fixture(
        ["kubernetes is kubernetes and kubernetes again here now"],
        gaps=1.0,
        drift_terms=_TERM,
    )


def _cfg(seq=0, **kw):
    return TranscriptionConfig(sample_rate=16000, channels=1, audio_format="pcm", seq=seq, **kw)


async def test_drift_client_drifts_without_carryover():
    d = _one_clip_drift()
    client = DriftDictationClient(d, {0: (0.0, d.fixture.total_s)})
    res = await client.transcribe(b"", _cfg())
    assert "koobernetes" in res.text  # rank-2 occurrence drifted
    assert res.text.split().count("kubernetes") == 2


async def test_drift_client_pins_with_keyterms():
    d = _one_clip_drift()
    client = DriftDictationClient(d, {0: (0.0, d.fixture.total_s)})
    res = await client.transcribe(b"", _cfg(keyterms_prompt=["kubernetes"]))
    assert "koobernetes" not in res.text
    assert res.text.split().count("kubernetes") == 3


async def test_drift_client_pins_with_stt_prompt():
    d = _one_clip_drift()
    client = DriftDictationClient(d, {0: (0.0, d.fixture.total_s)})
    res = await client.transcribe(b"", _cfg(stt_prompt="context: kubernetes matters"))
    assert "koobernetes" not in res.text


def test_terminology_consistency_rate_bounds():
    d = _one_clip_drift()
    assert terminology_consistency_rate("kubernetes kubernetes", d) == 1.0
    assert terminology_consistency_rate("koobernetes koobernetes", d) == 0.0


def test_paragraph_boundary_f1_cases():
    assert paragraph_boundary_f1([1.0, 2.0], [1.0, 2.0])[2] == 1.0
    assert paragraph_boundary_f1([5.0], [1.0])[2] == 0.0
    assert paragraph_boundary_f1([], [1.0])[2] == 0.0
    assert paragraph_boundary_f1([], [])[2] == 1.0


def _four_clip_drift():
    clip = (
        "alpha beta gamma delta kubernetes epsilon zeta eta theta "
        "iota kappa lambda mu nu xi omicron pi rho"
    )  # 18 words, kubernetes once per clip
    return build_drift_fixture([clip] * 4, gaps=1.5, drift_terms=_TERM)


async def test_compare_drift_three_conditions_and_charts():
    d = _four_clip_drift()
    a, b, c = await compare_drift(d, chop_s=5.0)

    # terminology: carryover + consistency pass lifts C to perfect; B drifts
    assert c.terminology_consistency == 1.0
    assert b.terminology_consistency < 1.0
    assert c.terminology_consistency > b.terminology_consistency

    # paragraph structure: A threw timing away -> 0; B/C recover it
    assert a.paragraph_f1 == 0.0
    assert b.paragraph_f1 >= 0.9
    assert c.paragraph_f1 == b.paragraph_f1

    # carryover never hurts WER
    assert c.wer <= b.wer


async def test_compare_drift_is_deterministic():
    d = _four_clip_drift()
    assert await compare_drift(d, chop_s=5.0) == await compare_drift(d, chop_s=5.0)


async def test_format_table_shows_three_charts():
    d = _four_clip_drift()
    results = await compare_drift(d, chop_s=5.0)
    table = format_table(results, d.fixture)
    assert "WER" in table
    assert "Terminology consistency" in table
    assert "Paragraph-boundary F1" in table
