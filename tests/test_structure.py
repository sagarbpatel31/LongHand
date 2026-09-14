"""structure.py: pause bands -> paragraphs/sections, gap markers, boundaries."""

from __future__ import annotations

from longhand.assemble.order import AssembledSegment
from longhand.assemble.structure import GAP_MARKER, build_document
from longhand.stt.fake import make_result


def _seg(seq: int, text: str, lead_pause: float, *, forced: bool = False) -> AssembledSegment:
    return AssembledSegment(
        seq=seq, result=make_result(text), lead_pause=lead_pause, forced=forced, is_gap=False
    )


def _gap(seq: int, lead_pause: float, error: str = "ServerError") -> AssembledSegment:
    return AssembledSegment(
        seq=seq, result=None, lead_pause=lead_pause, forced=False, is_gap=True, error=error
    )


def test_sub_paragraph_pause_stays_same_paragraph():
    doc = build_document([_seg(0, "a", 0.0), _seg(1, "b", 1.19)])
    assert doc.paragraph_boundaries == ()
    assert len(doc.sections) == 1
    assert len(doc.sections[0].paragraphs) == 1
    assert doc.sections[0].paragraphs[0].to_text() == "a b"


def test_paragraph_band_starts_new_paragraph():
    doc = build_document([_seg(0, "a", 0.0), _seg(1, "b", 1.2), _seg(2, "c", 0.5)])
    assert doc.paragraph_boundaries == (1,)  # only block 1 opens a paragraph
    assert doc.section_boundaries == ()
    assert len(doc.sections) == 1
    assert [p.to_text() for p in doc.sections[0].paragraphs] == ["a", "b c"]


def test_section_band_starts_new_section():
    doc = build_document([_seg(0, "a", 0.0), _seg(1, "b", 3.0)])
    assert doc.section_boundaries == (1,)
    assert 1 in doc.paragraph_boundaries  # section break is also a paragraph break
    assert len(doc.sections) == 2


def test_thresholds_are_inclusive():
    # exactly 1.2 -> paragraph; exactly 3.0 -> section
    doc = build_document([_seg(0, "a", 0.0), _seg(1, "b", 1.2), _seg(2, "c", 3.0)])
    assert doc.paragraph_boundaries == (1, 2)
    assert doc.section_boundaries == (2,)


def test_first_block_is_never_a_boundary():
    doc = build_document([_seg(0, "only", 5.0)])  # large lead_pause, still first
    assert doc.paragraph_boundaries == ()
    assert doc.section_boundaries == ()


def test_gap_renders_marker_and_is_never_dropped():
    doc = build_document([_seg(0, "a", 0.0), _gap(1, 0.5), _seg(2, "c", 0.5)])
    text = doc.to_text()
    assert GAP_MARKER in text
    assert "a" in text and "c" in text
    md = doc.to_markdown()
    assert GAP_MARKER in md and "seq=1" in md and "ServerError" in md


def test_gap_with_long_pause_still_opens_section():
    doc = build_document([_seg(0, "a", 0.0), _gap(1, 4.0)])
    assert doc.section_boundaries == (1,)
    assert len(doc.sections) == 2


def test_markdown_section_rule_and_paragraph_spacing():
    doc = build_document(
        [_seg(0, "a", 0.0), _seg(1, "b", 1.5), _seg(2, "c", 3.0)]
    )
    md = doc.to_markdown()
    assert "\n\n---\n\n" in md  # section rule
    assert md.count("\n\n") >= 2  # paragraph + section spacing
