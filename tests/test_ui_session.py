"""ReplaySession drives the condition-C pipeline for the UI, offline (§2).

Every assertion runs through the drift ASR client — no network, no mic. We check
the message contract the HTML page depends on: init shape, pending-before-document
ordering, the growing document snapshot, the verbatim/clean divergence that the
toggle shows, pause->structure, and the forced-cut seam inspector.
"""

from __future__ import annotations

from longhand.bench.fixtures import (
    DriftTerm,
    build_drift_fixture,
    demo_drift_fixture,
)
from longhand.ui.session import ReplaySession, _clean_blocks


async def _drain(session: ReplaySession) -> list[dict]:
    return [m async for m in session.stream()]


# --- message contract -------------------------------------------------------


async def test_init_message_first_and_describes_session():
    drift = demo_drift_fixture()
    session = ReplaySession(drift)
    msgs = await _drain(session)

    init = msgs[0]
    assert init["type"] == "init"
    assert init["n_segments"] == session.n_segments > 0
    assert init["paragraph_pause_s"] == 1.2
    assert init["section_pause_s"] == 3.0
    canons = {t["canonical"] for t in init["drift_terms"]}
    assert {"kubernetes", "ingress", "scheduler"} <= canons


async def test_pending_precedes_each_document_and_seqs_advance():
    drift = demo_drift_fixture()
    msgs = await _drain(ReplaySession(drift))

    # After init: strict (pending, document) pairs, one per segment, in seq order.
    body = msgs[1:]
    assert len(body) % 2 == 0
    seen_seq = -1
    for i in range(0, len(body), 2):
        pending, doc = body[i], body[i + 1]
        assert pending["type"] == "pending"
        assert doc["type"] == "document"
        assert pending["seq"] == doc["seq"] == seen_seq + 1
        seen_seq = doc["seq"]


async def test_final_document_is_complete_and_scored():
    drift = demo_drift_fixture()
    session = ReplaySession(drift)
    msgs = await _drain(session)

    docs = [m for m in msgs if m["type"] == "document"]
    assert docs[-1]["final"] is True
    assert all(d["final"] is False for d in docs[:-1])

    stats = docs[-1]["stats"]
    assert stats["segments_done"] == stats["segments_total"] == session.n_segments
    # Non-final snapshots defer the expensive scores; the final one carries them.
    assert stats["wer"] is not None
    assert stats["terminology_consistency"] is not None


async def test_document_snapshot_grows_monotonically():
    drift = demo_drift_fixture()
    docs = [m for m in await _drain(ReplaySession(drift)) if m["type"] == "document"]

    def n_blocks(doc: dict) -> int:
        return sum(
            len(p["blocks"]) for s in doc["sections"] for p in s["paragraphs"]
        )

    counts = [n_blocks(d) for d in docs]
    assert counts == sorted(counts)  # never shrinks
    assert counts[-1] == docs[-1]["stats"]["segments_done"]


# --- silence IS structure (idea #2) ----------------------------------------


async def test_pauses_become_paragraphs_and_sections():
    # demo gaps cycle 0.8 / 1.5 / 4.0 -> sentence / paragraph / section breaks.
    drift = demo_drift_fixture()
    final = [m for m in await _drain(ReplaySession(drift)) if m["type"] == "document"][-1]
    assert final["stats"]["sections"] > 1
    assert final["stats"]["paragraphs"] > final["stats"]["sections"]


# --- verbatim / clean toggle (condition C) ---------------------------------


async def test_carryover_makes_clean_diverge_from_verbatim_and_consistent():
    drift = demo_drift_fixture()
    final = [m for m in await _drain(ReplaySession(drift)) if m["type"] == "document"][-1]

    blocks = [b for s in final["sections"] for p in s["paragraphs"] for b in p["blocks"]]

    # The drift variants must survive in verbatim...
    verbatim_all = " ".join(b["verbatim"] for b in blocks)
    assert "koobernetes" in verbatim_all or "schedular" in verbatim_all or "ingres" in verbatim_all

    # ...but be normalised away in the cleaned view.
    clean_all = " ".join(b["clean"] for b in blocks)
    for bad in ("koobernetes", "schedular", "ingres "):
        assert bad not in clean_all

    # Condition C is fully consistent on this fixture (bench headline: 100%).
    assert final["stats"]["terminology_consistency"] == 1.0


def test_clean_blocks_preserves_per_block_token_counts():
    # The re-split trick is only valid if consistency_pass keeps token count.
    # Two 'kubernetes' vs one 'koobernetes' -> strict majority normalises the odd one.
    verbatims = ["the koobernetes cluster", "", "the kubernetes ingress kubernetes here"]
    cleans = _clean_blocks(verbatims)
    assert len(cleans) == len(verbatims)
    for v, c in zip(verbatims, cleans):
        assert len(c.split()) == len(v.split())
    assert "koobernetes" not in " ".join(cleans)


# --- forced-cut seam inspector (§6) ----------------------------------------


def _runon_drift_fixture():
    """One long run-on clip (no pauses) that trips forced cuts, with a drift term."""
    vocab = "the kubernetes operator reconciles the declared cluster state again".split()
    long_clip = " ".join(vocab[i % len(vocab)] for i in range(60))
    return build_drift_fixture(
        [long_clip],
        gaps=[0.0],
        drift_terms=[DriftTerm("kubernetes", "koobernetes")],
    )


async def test_forced_cut_emits_seam_with_deduplicated_overlap():
    drift = _runon_drift_fixture()
    # Small ceiling forces several cuts inside the single run-on clip.
    session = ReplaySession(drift, max_segment_s=5.0)
    assert session.forced_cut_rate > 0

    final = [m for m in await _drain(session) if m["type"] == "document"][-1]
    blocks = [b for s in final["sections"] for p in s["paragraphs"] for b in p["blocks"]]

    seams = [b["seam"] for b in blocks if b["seam"] is not None]
    assert seams, "a forced cut must produce a seam on the following block"
    for seam in seams:
        # The spliced result drops the duplicated overlap: fewer tokens than the
        # naive concat whenever an alignment was found.
        if seam["aligned"]:
            assert seam["overlap_len"] > 0
            assert len(seam["spliced"].split()) < len(seam["raw_concat"].split())


async def test_no_forced_cut_means_no_seams():
    drift = demo_drift_fixture()  # every cut lands in silence -> zero forced
    session = ReplaySession(drift)
    assert session.forced_cut_rate == 0.0
    final = [m for m in await _drain(session) if m["type"] == "document"][-1]
    blocks = [b for s in final["sections"] for p in s["paragraphs"] for b in p["blocks"]]
    assert all(b["seam"] is None for b in blocks)


async def test_stream_is_deterministic():
    drift = demo_drift_fixture()
    a = await _drain(ReplaySession(drift))
    b = await _drain(ReplaySession(drift))
    assert a == b
