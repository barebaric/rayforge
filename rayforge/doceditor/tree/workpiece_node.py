"""The row for one workpiece, with the steps that operate on it."""

from collections.abc import Set as AbstractSet

from gi.repository import Gtk

from ...core.workflow import Workflow
from ...core.workpiece import WorkPiece
from .node import NodeRow
from .specs import NodeSpec
from .step_node import steps_for

_WORKPIECE_ICON = "model-symbolic"


def _workflow_for(workpiece: WorkPiece) -> Workflow | None:
    """The first workflow of the part's layer that runs for it."""
    layer = workpiece.layer
    if layer is None:
        return None
    workpieces = layer.all_workpieces
    for workflow in layer.workflows:
        if workpiece in workflow.selected_workpieces(workpieces):
            return workflow
    return None


class WorkPieceNode(NodeRow):
    """The row for one part, optionally followed by its steps."""

    def __init__(
        self,
        workpiece: WorkPiece,
        depth: int = 1,
        expanded: AbstractSet[str] = frozenset(),
        selected: AbstractSet[str] = frozenset(),
        callbacks=None,
    ):
        super().__init__(
            self.build_spec(workpiece, depth, expanded, selected), callbacks
        )

    @staticmethod
    def build_spec(
        workpiece: WorkPiece,
        depth: int = 1,
        expanded: AbstractSet[str] = frozenset(),
        selected: AbstractSet[str] = frozenset(),
    ) -> NodeSpec:
        workflow = _workflow_for(workpiece)
        steps = steps_for(workpiece, workflow)
        return NodeSpec(
            kind="workpiece",
            uid=workpiece.uid,
            depth=depth,
            label=workpiece.name,
            icon=_WORKPIECE_ICON,
            detail=_size_detail(workpiece),
            eye=True,
            on=workpiece.visible,
            selected=workpiece.uid in selected,
            has_children=bool(steps),
            expanded=workpiece.uid in expanded,
        )

    @classmethod
    def build_rows(
        cls,
        workpiece: WorkPiece,
        depth: int,
        expanded: AbstractSet[str],
        selected: AbstractSet[str],
        callbacks,
    ) -> list[Gtk.Widget]:
        from .step_node import StepNode

        rows: list[Gtk.Widget] = [
            cls(workpiece, depth, expanded, selected, callbacks)
        ]
        if workpiece.uid not in expanded:
            return rows
        workflow = _workflow_for(workpiece)
        for step in steps_for(workpiece, workflow):
            rows.append(StepNode(step, depth + 1, callbacks))
        return rows


def _size_detail(workpiece: WorkPiece) -> str:
    width = workpiece.natural_width_mm
    height = workpiece.natural_height_mm
    if width <= 0 and height <= 0:
        return ""
    return f"{width:.0f} × {height:.0f} mm"
