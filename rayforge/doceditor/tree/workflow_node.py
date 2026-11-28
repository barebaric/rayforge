"""The row for one workflow with the parts it operates on.

In the current model a layer owns exactly one workflow, so the row is
built from the layer and shows the layer's name and color.
"""

from collections.abc import Set as AbstractSet
from gettext import ngettext

from gi.repository import Gtk

from ...core.layer import Layer
from .node import NodeRow
from .specs import NodeSpec


class WorkflowNode(NodeRow):
    """The row for one workflow, optionally followed by its parts."""

    def __init__(
        self,
        layer: Layer,
        depth: int = 1,
        expanded: AbstractSet[str] = frozenset(),
        selected: AbstractSet[str] = frozenset(),
        callbacks=None,
    ):
        super().__init__(
            self.build_spec(layer, depth, expanded, selected), callbacks
        )

    @staticmethod
    def build_spec(
        layer: Layer,
        depth: int = 1,
        expanded: AbstractSet[str] = frozenset(),
        selected: AbstractSet[str] = frozenset(),
    ) -> NodeSpec:
        workflow = layer.workflow
        step_count = (
            len([step for step in workflow.steps if step.visible])
            if workflow
            else 0
        )
        return NodeSpec(
            kind="workflow",
            uid=workflow.uid if workflow else layer.uid,
            depth=depth,
            label=layer.name,
            color=layer.color,
            detail=ngettext("{n} step", "{n} steps", step_count).format(
                n=step_count
            ),
            selected=workflow.uid in selected if workflow else False,
            has_children=bool(layer.all_workpieces) and workflow is not None,
            expanded=workflow.uid in expanded if workflow else False,
        )

    @classmethod
    def build_rows(
        cls,
        layer: Layer,
        depth: int,
        expanded: AbstractSet[str],
        selected: AbstractSet[str],
        callbacks,
    ) -> list[Gtk.Widget]:
        from .part_node import PartNode

        workflow = layer.workflow
        rows: list[Gtk.Widget] = [
            cls(layer, depth, expanded, selected, callbacks)
        ]
        if workflow is None or workflow.uid not in expanded:
            return rows
        for number, workpiece in enumerate(layer.all_workpieces, start=1):
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
