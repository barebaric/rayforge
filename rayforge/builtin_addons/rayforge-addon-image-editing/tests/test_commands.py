from image_editing.commands import (
    replace_op_edits,
    set_op_edits,
    toggle_image_edit,
)
from image_editing.ops import INVERT

from rayforge.core.image_edit import ImageEditStep
from rayforge.core.undo import HistoryManager
from rayforge.core.workpiece import WorkPiece

OTHER = ImageEditStep("other")


def test_adds_the_edit_to_every_workpiece():
    a, b = WorkPiece("a"), WorkPiece("b")
    b.image_edits = [OTHER]
    assert toggle_image_edit(HistoryManager(), [a, b], INVERT, "Invert")
    assert a.image_edits == (INVERT,)
    assert b.image_edits == (OTHER, INVERT)


def test_removes_the_edit_when_every_workpiece_has_it():
    a, b = WorkPiece("a"), WorkPiece("b")
    a.image_edits = [INVERT]
    b.image_edits = [OTHER, INVERT]
    assert not toggle_image_edit(HistoryManager(), [a, b], INVERT, "Invert")
    assert a.image_edits == ()
    assert b.image_edits == (OTHER,)


def test_mixed_selection_adds_where_missing():
    a, b = WorkPiece("a"), WorkPiece("b")
    a.image_edits = [INVERT]
    assert toggle_image_edit(HistoryManager(), [a, b], INVERT, "Invert")
    assert a.image_edits == (INVERT,)
    assert b.image_edits == (INVERT,)


def test_one_undo_step_restores_everything():
    history = HistoryManager()
    a, b = WorkPiece("a"), WorkPiece("b")
    toggle_image_edit(history, [a, b], INVERT, "Invert")
    history.undo()
    assert a.image_edits == ()
    assert b.image_edits == ()
    history.redo()
    assert a.image_edits == (INVERT,)
    assert b.image_edits == (INVERT,)


def test_empty_selection_does_nothing():
    history = HistoryManager()
    assert not toggle_image_edit(history, [], INVERT, "Invert")
    assert not history.can_undo()


TONE = ImageEditStep("tone", {"brightness": 10})
NEW_TONE = ImageEditStep("tone", {"brightness": 20})
SHARPEN = ImageEditStep("sharpen", {"amount": 50, "radius_mm": 0.2})


def test_replace_keeps_the_position_of_the_op():
    edits = (TONE, INVERT)
    assert replace_op_edits(edits, {"tone": NEW_TONE}) == (NEW_TONE, INVERT)


def test_replace_appends_missing_ops_in_order():
    result = replace_op_edits((INVERT,), {"sharpen": SHARPEN, "tone": TONE})
    assert result == (INVERT, SHARPEN, TONE)


def test_replace_with_none_removes_the_op():
    edits = (TONE, INVERT, SHARPEN)
    result = replace_op_edits(edits, {"tone": None, "sharpen": None})
    assert result == (INVERT,)


def test_replace_drops_duplicate_edits_of_the_op():
    edits = (TONE, INVERT, TONE)
    assert replace_op_edits(edits, {"tone": NEW_TONE}) == (NEW_TONE, INVERT)


def test_set_op_edits_is_one_undo_step():
    history = HistoryManager()
    a, b = WorkPiece("a"), WorkPiece("b")
    b.image_edits = [INVERT]
    set_op_edits(history, [a, b], {"tone": TONE}, "Adjust")
    assert a.image_edits == (TONE,)
    assert b.image_edits == (INVERT, TONE)
    history.undo()
    assert a.image_edits == ()
    assert b.image_edits == (INVERT,)


def test_set_op_edits_undoes_to_the_stack_before_the_preview():
    history = HistoryManager()
    wp = WorkPiece("a")
    wp.image_edits = [NEW_TONE]
    set_op_edits(history, [wp], {"tone": TONE}, "Adjust", {wp.uid: ()})
    assert wp.image_edits == (TONE,)
    history.undo()
    assert wp.image_edits == ()


def test_set_op_edits_without_change_restores_the_preview():
    history = HistoryManager()
    wp = WorkPiece("a")
    wp.image_edits = [NEW_TONE]
    set_op_edits(history, [wp], {"tone": None}, "Adjust", {wp.uid: ()})
    assert wp.image_edits == ()
    assert not history.can_undo()
