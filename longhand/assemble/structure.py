"""Silence IS structure (project idea #2).

The VAD gives us pause durations for free; naive chunking throws them away. Here
we turn each segment's `lead_pause` (the silence before it, recorded by the cut
policy) into document structure:

    < paragraph_pause   -> sentence break, same paragraph
    paragraph..section  -> new paragraph
    >= section_pause    -> new section (also a paragraph boundary)

Gaps (permanently failed segments) render as a visible marker and still carry
structure via their pause — audio is never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

from .order import AssembledSegment

GAP_MARKER = "[audio unavailable]"

PARAGRAPH_PAUSE_S = 1.2
SECTION_PAUSE_S = 3.0


@dataclass(frozen=True)
class Block:
    seq: int
    text: str  # best_text; "" for a gap
    lead_pause: float
    is_gap: bool
    forced: bool
    error: str | None = None

    def render(self, *, markdown: bool = False) -> str:
        if self.is_gap:
            if markdown and self.error:
                return f"{GAP_MARKER} (seq={self.seq}: {self.error})"
            return GAP_MARKER
        return self.text


@dataclass(frozen=True)
class Paragraph:
    blocks: tuple[Block, ...]

    def to_text(self) -> str:
        return " ".join(b.render() for b in self.blocks)

    def to_markdown(self) -> str:
        return " ".join(b.render(markdown=True) for b in self.blocks)


@dataclass(frozen=True)
class Section:
    paragraphs: tuple[Paragraph, ...]


@dataclass(frozen=True)
class Document:
    sections: tuple[Section, ...]
    paragraph_boundaries: tuple[int, ...]  # block indices that open a new paragraph/section
    section_boundaries: tuple[int, ...]  # block indices that open a new section

    def to_text(self) -> str:
        paras: list[str] = []
        for section in self.sections:
            paras.extend(p.to_text() for p in section.paragraphs)
        return "\n\n".join(paras)

    def to_markdown(self) -> str:
        rendered_sections: list[str] = []
        for section in self.sections:
            rendered_sections.append("\n\n".join(p.to_markdown() for p in section.paragraphs))
        return "\n\n---\n\n".join(rendered_sections)


def _classify(lead_pause: float, paragraph_pause: float, section_pause: float) -> str:
    if lead_pause >= section_pause:
        return "section"
    if lead_pause >= paragraph_pause:
        return "paragraph"
    return "same"


def build_document(
    segments: list[AssembledSegment],
    *,
    paragraph_pause: float = PARAGRAPH_PAUSE_S,
    section_pause: float = SECTION_PAUSE_S,
) -> Document:
    """Group assembled segments into sections/paragraphs from their pauses."""
    sections: list[list[list[Block]]] = []  # sections -> paragraphs -> blocks
    paragraph_boundaries: list[int] = []
    section_boundaries: list[int] = []

    for i, seg in enumerate(segments):
        block = Block(
            seq=seg.seq,
            text=seg.best_text,
            lead_pause=seg.lead_pause,
            is_gap=seg.is_gap,
            forced=seg.forced,
            error=seg.error,
        )
        if i == 0:
            sections.append([[block]])  # first block: section 0 / paragraph 0
            continue

        kind = _classify(seg.lead_pause, paragraph_pause, section_pause)
        if kind == "section":
            sections.append([[block]])
            section_boundaries.append(i)
            paragraph_boundaries.append(i)  # a section break is also a paragraph break
        elif kind == "paragraph":
            sections[-1].append([block])
            paragraph_boundaries.append(i)
        else:  # same paragraph
            sections[-1][-1].append(block)

    built = tuple(
        Section(tuple(Paragraph(tuple(blocks)) for blocks in paragraphs))
        for paragraphs in sections
    )
    return Document(
        sections=built,
        paragraph_boundaries=tuple(paragraph_boundaries),
        section_boundaries=tuple(section_boundaries),
    )
