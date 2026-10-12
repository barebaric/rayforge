from image_editing.commands import toggle_image_edit
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
