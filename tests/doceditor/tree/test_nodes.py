"""Tests for the per-node spec builders."""

import pytest

from rayforge.core.doc import Doc
from rayforge.core.group import Group
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.tree.group_node import GroupNode
from rayforge.doceditor.tree.part_node import PartNode
from rayforge.doceditor.tree.step_node import StepNode, steps_for
from rayforge.doceditor.tree.stock_node import StockNode
from rayforge.doceditor.tree.workflow_node import WorkflowNode
from rayforge.doceditor.tree.workpiece_node import WorkPieceNode


class CutStep(Step):
    TYPELABEL = "Contour"
    ICON = "cut-symbolic"

    def __init__(self, name: str | None = None):
        super().__init__(typelabel=self.TYPELABEL, name=name)


class EngraveStep(Step):
    TYPELABEL = "Engrave"
    ICON = "engrave-symbolic"

    def __init__(self, name: str | None = None):
        super().__init__(typelabel=self.TYPELABEL, name=name)


@pytest.fixture
def doc() -> Doc:
    doc = Doc()
    doc.set_children([])

    parts = Layer("Parts")
    doc.add_layer(parts)
    parts_workflow = parts.workflows[0]
    ribs = Group("Ribs")
    parts.add_child(ribs)
    rib_left = WorkPiece("Rib L")
    rib_left.natural_width_mm = 70
    rib_left.natural_height_mm = 40
    ribs.add_child(rib_left)
    rib_right = WorkPiece("Rib R")
    ribs.add_child(rib_right)
    fuselage = WorkPiece("Fuselage")
    fuselage.natural_width_mm = 130
    fuselage.natural_height_mm = 60
    parts.add_child(fuselage)
    parts_workflow.add_step(CutStep(name="Contour outer"))
    inner = CutStep(name="Contour inner")
    inner.workpiece_uids = [rib_left.uid, rib_right.uid]
    parts_workflow.add_step(inner)

    names = Layer("Names")
    doc.add_layer(names)
    names_workflow = names.workflows[0]
    name = WorkPiece("Fuselage name")
    names.add_child(name)
    names_workflow.add_step(EngraveStep(name="Engrave"))

    asset = StockAsset(name="Balsa 3 mm")
    asset.thickness = 3.0
    doc.add_asset(asset)
    doc.layers[0].add_child(
        StockItem(stock_asset_uid=asset.uid, name="Balsa 3 mm")
    )
    return doc


def test_stock_spec(doc):
    stock = doc.stock_items[0]
    spec = StockNode.build_spec(stock, 1, set())
    assert spec.kind == "stock"
    assert spec.label == "Balsa 3 mm"
    assert spec.detail == "3.0 mm"
    assert spec.eye is True and spec.on is True
    assert spec.has_children is False

    stock.visible = False
    spec = StockNode.build_spec(stock, 1, set())
    assert spec.on is False


def test_stock_spec_selection(doc):
    stock = doc.stock_items[0]
    spec = StockNode.build_spec(stock, 1, {stock.uid})
    assert spec.selected is True


def test_group_spec(doc):
    group = doc.layers[0].content_items[0]
    spec = GroupNode.build_spec(group, 1)
    assert spec.kind == "group"
    assert spec.label == "Ribs"
    assert spec.detail == "2 parts"
    assert spec.eye is True and spec.on is True
    assert spec.has_children is True
    assert spec.expanded is False

    spec = GroupNode.build_spec(group, 1, expanded={group.uid})
    assert spec.expanded is True

    group.visible = False
    assert GroupNode.build_spec(group, 1).on is False


def test_workpiece_spec(doc):
    workpiece = doc.layers[0].all_workpieces[0]
    spec = WorkPieceNode.build_spec(workpiece, 2)
    assert spec.kind == "workpiece"
    assert spec.label == "Rib L"
    assert spec.detail == "70 × 40 mm"
    assert spec.eye is True and spec.on is True
    assert spec.has_children is False
    assert spec.expanded is False

    spec = WorkPieceNode.build_spec(workpiece, 2, expanded={workpiece.uid})
    assert spec.expanded is True

    workpiece.visible = False
    assert WorkPieceNode.build_spec(workpiece, 2).on is False


def test_steps_for_respects_step_scoping(doc):
    parts_layer = doc.layers[0]
    rib_left = parts_layer.all_workpieces[0]
    fuselage = parts_layer.all_workpieces[-1]
    workflow = parts_layer.workflows[0]
    assert workflow is not None

    assert [s.name for s in steps_for(rib_left, workflow)] == [
        "Contour outer",
        "Contour inner",
    ]
    # Fuselage is not in the inner step's scope, so it only gets the
    # unscoped outer step.
    assert [s.name for s in steps_for(fuselage, workflow)] == ["Contour outer"]

    hidden = CutStep(name="Hidden")
    hidden.visible = False
    workflow.add_step(hidden)
    assert [s.name for s in steps_for(rib_left, workflow)] == [
        "Contour outer",
        "Contour inner",
    ]


def test_step_spec(doc):
    parts_workflow = doc.layers[0].workflows[0]
    step = parts_workflow.steps[0]
    spec = StepNode.build_spec(step, 2)
    assert spec.kind == "step"
    assert spec.label == "Contour outer"
    assert spec.icon == "cut-symbolic"
    assert spec.detail == "Contour"
    assert spec.on is True

    step.visible = False
    assert StepNode.build_spec(step, 2).on is False


def test_workflow_spec(doc):
    parts = doc.layers[0]
    workflow = parts.workflows[0]
    spec = WorkflowNode.build_spec(workflow, 1)
    assert spec.kind == "workflow"
    assert spec.label == "Workflow 1"
    assert spec.color == workflow.color
    assert spec.detail == "2 steps"
    assert spec.has_children is True
    assert spec.expanded is False

    spec = WorkflowNode.build_spec(workflow, 1, expanded={workflow.uid})
    assert spec.expanded is True


def test_part_spec(doc):
    parts = doc.layers[0]
    workflow = parts.workflows[0]
    rib_left = parts.all_workpieces[0]
    fuselage = parts.all_workpieces[-1]

    spec = PartNode.build_spec(rib_left, 1, workflow, 2)
    assert spec.kind == "part"
    assert spec.label == "Rib L"
    assert spec.badge == "1"
    assert spec.detail == "2 steps"

    spec = PartNode.build_spec(fuselage, 3, workflow, 2)
    assert spec.badge == "3"
    assert spec.detail == "1 step"

    fuselage.visible = False
    assert PartNode.build_spec(fuselage, 3, workflow, 2).on is False


def test_part_spec_selection(doc):
    parts = doc.layers[0]
    workflow = parts.workflows[0]
    rib_left = parts.all_workpieces[0]
    spec = PartNode.build_spec(rib_left, 1, workflow, 2, {rib_left.uid})
    assert spec.selected is True


def test_workpiece_feature_rows(doc):
    from raygeo.geo import Geometry

    workpiece = doc.layers[0].all_workpieces[0]
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

    spec = WorkPieceNode.build_spec(workpiece, 2)
    assert spec.has_children is True

    from rayforge.doceditor.tree.node import NodeRow

    rows = [
        row
        for row in WorkPieceNode.build_rows(
            workpiece, 2, {workpiece.uid}, set(), None
        )
        if isinstance(row, NodeRow)
    ]
    labels = [row.spec.label for row in rows if row.spec.kind == "feature"]
    assert "Outer paths" in labels
    assert "Inner paths" in labels
    assert all(
        row.spec.selectable is False
        for row in rows
        if row.spec.kind == "feature"
    )
