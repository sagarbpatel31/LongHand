"""OrderedAssembler: ordering (§9.4) and permanent-failure gap markers (§9.6)."""

from __future__ import annotations

import pytest

from longhand.assemble.order import AssemblyError, OrderedAssembler
from longhand.stt.fake import make_result


def _add(asm: OrderedAssembler, seq: int, *, lead_pause: float = 0.0, forced: bool = False):
    asm.add_result(seq, make_result(f"seg{seq}"), lead_pause=lead_pause, forced=forced)


def test_in_order_arrival_emits_immediately():
    asm = OrderedAssembler()
    _add(asm, 0)
    assert [s.seq for s in asm.ready()] == [0]
    _add(asm, 1)
    assert [s.seq for s in asm.ready()] == [1]


def test_reverse_order_arrival_assembles_correctly():
    # §9.4 — responses returned in reverse order still assemble in seq order.
    asm = OrderedAssembler()
    for seq in (3, 2, 1):
        _add(asm, seq)
        assert asm.ready() == []  # blocked until seq 0 lands
    _add(asm, 0)
    assert [s.seq for s in asm.ready()] == [0, 1, 2, 3]


def test_interleaved_hole_blocks_then_flushes():
    asm = OrderedAssembler()
    _add(asm, 0)
    _add(asm, 2)
    assert [s.seq for s in asm.ready()] == [0]  # seq 1 hole blocks 2
    _add(asm, 1)
    assert [s.seq for s in asm.ready()] == [1, 2]


def test_permanent_failure_becomes_visible_gap_neighbors_intact():
    # §9.6 — one segment fails permanently; document survives with a gap marker.
    asm = OrderedAssembler()
    _add(asm, 0)
    asm.add_gap(1, lead_pause=1.5, forced=False, error="ServerError")
    _add(asm, 2)
    doc = asm.emit_all()

    assert [s.seq for s in doc] == [0, 1, 2]
    gap = doc[1]
    assert gap.is_gap is True
    assert gap.result is None
    assert gap.error == "ServerError"
    assert gap.best_text == ""
    assert gap.lead_pause == 1.5
    # neighbors intact
    assert doc[0].best_text == "seg0"
    assert doc[2].best_text == "seg2"


def test_emit_all_raises_on_hole():
    asm = OrderedAssembler()
    _add(asm, 0)
    _add(asm, 2)  # seq 1 missing
    with pytest.raises(AssemblyError):
        asm.emit_all()


def test_add_is_idempotent_per_seq():
    # A retry double-reporting the same seq must not duplicate it.
    asm = OrderedAssembler()
    _add(asm, 0)
    _add(asm, 0)
    asm.add_gap(0, lead_pause=0.0, forced=False, error="late")  # ignored
    assert [s.seq for s in asm.emit_all()] == [0]


def test_lead_pause_and_forced_flags_carried_through():
    asm = OrderedAssembler()
    asm.add_result(0, make_result("a"), lead_pause=0.0, forced=False)
    asm.add_result(1, make_result("b"), lead_pause=2.4, forced=True)
    doc = asm.emit_all()
    assert doc[1].lead_pause == 2.4
    assert doc[1].forced is True
