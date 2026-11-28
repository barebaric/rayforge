"""The row for one workflow with the parts it operates on."""

from collections.abc import Set as AbstractSet
from gettext import ngettext

from gi.repository import Gtk

from ...core.setup import Setup
from ...core.workflow import Workflow
from .node import NodeRow
from .specs import NodeSpec


class WorkflowNode(NodeRow):
    """The row for one workflow, optionally followed by its parts."""

    def __init__(
        self,
        workflow: Workflow,
        depth: int = 1,
        expanded: AbstractSet[str] = frozenset(),
        selected: AbstractSet[str] = frozenset(),
        callbacks=None,
    ):
        super().__init__(
            self.build_spec(workflow, depth, expanded, selected), callbacks
        )

    @staticmethod
    def build_spec(
        workflow: Workflow,
        depth: int = 1,
        expanded: AbstractSet[str] = frozenset(),
        selected: AbstractSet[str] = frozenset(),
    ) -> NodeSpec:
        step_count = len([step for step in workflow.steps if step.visible])
        workpieces: list = []
        setup = workflow.parent
        if isinstance(setup, Setup):
            workpieces = workflow.selected_workpieces(setup.all_workpieces)
        return NodeSpec(
            kind="workflow",
            uid=workflow.uid,
            depth=depth,
            label=workflow.name,
            color=workflow.color,
            detail=ngettext("{n} step", "{n} steps", step_count).format(
                n=step_count
            ),
            selected=workflow.uid in selected,
            has_children=bool(workpieces),
            expanded=workflow.uid in expanded,
        )

    @classmethod
    def build_rows(
        cls,
        workflow: Workflow,
        depth: int,
        expanded: AbstractSet[str],
        selected: AbstractSet[str],
        callbacks,
    ) -> list[Gtk.Widget]:
        from .part_node import PartNode

        rows: list[Gtk.Widget] = [
            cls(workflow, depth, expanded, selected, callbacks)
        ]
        if workflow.uid not in expanded:
            return rows
        setup = workflow.parent
        if not isinstance(setup, Setup):
            return rows
        workpieces = workflow.selected_workpieces(setup.all_workpieces)
        for number, workpiece in enumerate(workpieces, start=1):
            rows.append(
                PartNode(
                    workpiece,
                    number,
                    workflow,
                    depth + 1,
                    selected,
                    callbacks,
                )
            )
        return rows
