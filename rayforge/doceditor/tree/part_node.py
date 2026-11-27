"""The row for one part as it appears beneath a workflow."""

from collections.abc import Set as AbstractSet
from gettext import ngettext

from ...core.workflow import Workflow
from ...core.workpiece import WorkPiece
from .node import NodeRow
from .specs import NodeSpec
from .step_node import steps_for


class PartNode(NodeRow):
    """The row for one part under a workflow, showing what the
    workflow's steps take from it."""

    def __init__(
        self,
        workpiece: WorkPiece,
        number: int,
        workflow: Workflow,
        depth: int = 2,
        selected: AbstractSet[str] = frozenset(),
        callbacks=None,
    ):
        super().__init__(
            self.build_spec(workpiece, number, workflow, depth, selected),
            callbacks,
        )

    @staticmethod
    def build_spec(
        workpiece: WorkPiece,
        number: int,
        workflow: Workflow,
        depth: int = 2,
        selected: AbstractSet[str] = frozenset(),
    ) -> NodeSpec:
        steps = steps_for(workpiece, workflow)
        detail = (
            ngettext("{n} step", "{n} steps", len(steps)).format(n=len(steps))
            if steps
            else ""
        )
        return NodeSpec(
            kind="part",
            uid=workpiece.uid,
            depth=depth,
            label=workpiece.name,
            badge=str(number),
            detail=detail,
            on=workpiece.visible,
            selected=workpiece.uid in selected,
        )
