"""Glossary: term extraction, keyterm packing, stt_prompt, consistency pass (§5)."""

from __future__ import annotations

from longhand.assemble.glossary import (
    Glossary,
    GlossaryConfig,
    consistency_pass,
    extract_terms,
)
from longhand.stt.client import TranscriptionConfig


# --- extraction -------------------------------------------------------------


def test_extract_finds_capitalized_proper_nouns():
    terms = extract_terms("We deployed Kubernetes and Envoy today")
    assert "Kubernetes" in terms
    assert "Envoy" in terms
    assert "We" not in terms  # sentence-initial capital ignored


def test_extract_works_on_lowercase_asr():
    terms = extract_terms("the kubernetes cluster reconciles state")
    lowered = {t.lower() for t in terms}
    assert "kubernetes" in lowered
    assert "the" not in lowered
    assert "state" not in lowered  # common word


def test_extract_ignores_all_common_words():
    assert extract_terms("the and of to in it is on at by") == []


# --- keyterms ---------------------------------------------------------------


def test_keyterms_respects_term_cap():
    g = Glossary()
    for i in range(150):
        g.observe(f"term{i}")  # digit-mixed -> a term
    assert len(g.keyterms()) <= 100


def test_keyterms_respects_char_cap():
    g = Glossary()
    for i in range(100):
        g.observe(f"term{i}" + "a" * 90)  # ~95-char distinct terms
    assert sum(len(t) for t in g.keyterms()) <= 8000


def test_keyterms_ranked_by_count():
    g = Glossary()
    for _ in range(5):
        g.observe("kubernetes")
    g.observe("envoyproxy")
    assert g.keyterms()[0].lower() == "kubernetes"


def test_keyterms_output_passes_config_validate():
    g = Glossary()
    for _ in range(3):
        g.observe("kubernetes envoyproxy reconciler")
    cfg = TranscriptionConfig(
        sample_rate=16000, channels=1, keyterms_prompt=g.keyterms()
    )
    cfg.validate()  # must not raise


# --- stt_prompt -------------------------------------------------------------


def test_stt_prompt_bounded_and_ends_with_tail():
    g = Glossary(config=GlossaryConfig(tail_chars=50))
    g.observe("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda")
    p = g.stt_prompt()
    assert len(p) <= 6000
    assert "Technical dictation" in p
    assert p.rstrip().endswith("lambda")


# --- consistency pass -------------------------------------------------------


def test_consistency_normalizes_minority_variant():
    text = " ".join(["kubernetes"] * 8 + ["coobernetes"])
    out = consistency_pass(text)
    assert "coobernetes" not in out.lower()
    assert out.lower().split().count("kubernetes") == 9


def test_consistency_leaves_distinct_terms_untouched():
    text = "cluster custard cluster custard"
    out = consistency_pass(text)
    assert "cluster" in out and "custard" in out
    assert out.split().count("cluster") == 2
    assert out.split().count("custard") == 2


def test_consistency_no_strict_majority_is_noop():
    text = "kubernetes coobernetes"  # 1 vs 1 -> no majority
    assert consistency_pass(text) == text


def test_consistency_is_idempotent():
    text = " ".join(["kubernetes"] * 8 + ["coobernetes"])
    once = consistency_pass(text)
    assert consistency_pass(once) == once


def test_consistency_preserves_punctuation():
    text = "kubernetes kubernetes kubernetes coobernetes, works"
    out = consistency_pass(text)
    assert "kubernetes," in out  # trailing comma preserved on the rewritten token
