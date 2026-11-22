from rayforge.core.doc import Doc
from rayforge.core.image_edit import ImageEditStep
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece

INVERT = ImageEditStep("test_invert")


class _LegacyStep(Step):
    def __init__(self, edits):
        super().__init__(typelabel="Legacy")
        self._legacy = list(edits)

    def pop_legacy_image_edits(self):
        edits, self._legacy = self._legacy, []
        return edits


def _doc_with(step: Step, *workpieces: WorkPiece) -> Doc:
    doc = Doc()
    for wp in workpieces:
        doc.active_layer.add_child(wp)
    workflow = doc.active_layer.workflow
    assert workflow is not None
    workflow.add_step(step)
    return doc


def test_base_step_has_no_legacy_image_edits():
    assert Step(typelabel="Plain").pop_legacy_image_edits() == []


def test_legacy_edits_move_to_the_layers_workpieces():
    a, b = WorkPiece("a"), WorkPiece("b")
    doc = _doc_with(_LegacyStep([INVERT]), a, b)
    doc.adopt_legacy_image_edits()
    assert a.image_edits == (INVERT,)
    assert b.image_edits == (INVERT,)


def test_other_layers_are_untouched():
    inside, outside = WorkPiece("in"), WorkPiece("out")
    doc = _doc_with(_LegacyStep([INVERT]), inside)
    other = Layer("Other")
    doc.add_layer(other)
    other.add_child(outside)
    doc.adopt_legacy_image_edits()
    assert outside.image_edits == ()


def test_adoption_is_idempotent():
    wp = WorkPiece("a")
    doc = _doc_with(_LegacyStep([INVERT]), wp)
    doc.adopt_legacy_image_edits()
    doc.adopt_legacy_image_edits()
    assert wp.image_edits == (INVERT,)


def test_an_edit_already_present_is_not_duplicated():
    wp = WorkPiece("a")
    wp.image_edits = [INVERT]
    doc = _doc_with(_LegacyStep([INVERT]), wp)
    doc.adopt_legacy_image_edits()
    assert wp.image_edits == (INVERT,)


def test_missing_image_edit_ops():
    wp = WorkPiece("a")
    wp.image_edits = [ImageEditStep("not_installed")]
    doc = _doc_with(Step(typelabel="Plain"), wp)
    assert doc.missing_image_edit_ops == {"not_installed"}
