"""The row for one step of a workflow."""

from ...core.step import Step
from ...core.workpiece import WorkPiece
from .node import NodeRow
from .specs import NodeSpec

_STEP_FALLBACK_ICON = "cut-symbolic"


def steps_for(workpiece: WorkPiece, workflow) -> list[Step]:
    """The workflow's visible steps that operate on the workpiece."""
    if workflow is None:
        return []
    return [
        step
        for step in workflow.steps
        if step.visible
        and (not step.workpiece_uids or workpiece.uid in step.workpiece_uids)
    ]


class StepNode(NodeRow):
    """The row for one step, as shown beneath a part."""

    def __init__(self, step: Step, depth: int = 2, callbacks=None):
        super().__init__(self.build_spec(step, depth), callbacks)

    @staticmethod
    def build_spec(step: Step, depth: int = 2) -> NodeSpec:
        return NodeSpec(
            kind="step",
            uid=step.uid,
            depth=depth,
            label=step.name,
            icon=step.ICON or _STEP_FALLBACK_ICON,
            detail=step.typelabel,
            on=step.visible,
        )
