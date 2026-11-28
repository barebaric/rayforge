"""UI tests for the Parts tree rows."""

import pytest
from raygeo.geo import Geometry

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.tree.parts_tree import PartsTree


@pytest.mark.ui
def test_workpiece_expands_into_feature_rows(ui_context_initializer):
    layer = Layer("Layer")
    workpiece = WorkPiece("Part")
    layer.add_child(workpiece)
    geometry = Geometry()
    geometry.move_to(0, 0)
    geometry.line_to(10, 0)
    geometry.line_to(10, 10)
    geometry.line_to(0, 10)
    geometry.close_path()
    inner = Geometry()
    inner.move_to(2, 2)
    inner.line_to(4, 2)
    inner.line_to(4, 4)
    inner.close_path()
    geometry.extend(inner)
    workpiece._edited_boundaries = geometry

    doc = Doc()
    doc.set_children([])
    doc.add_layer(layer)
    tree = PartsTree()
    tree.set_doc(doc)
    tree.set_layer(layer)
    tree.expand(workpiece.uid)

    features = [row for row in tree.rows() if row.spec.kind == "feature"]
    labels = [row.spec.label for row in features]
    assert "Outer paths" in labels
    assert "Inner paths" in labels
    assert all(not row.spec.selectable for row in features)
