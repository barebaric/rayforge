"""
Defines the Workflow class, which holds an ordered sequence of Steps.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypeVar

from blinker import Signal
from raygeo.geo import Matrix

from .color import COLOR_PALETTE
from .item import DocItem
from .step import Step
from .workpiece import WorkPiece

if TYPE_CHECKING:
    from .layer import Layer

logger = logging.getLogger(__name__)

# For generic type hinting in add_child
T = TypeVar("T", bound="DocItem")


class Workflow(DocItem):
    """
    An ordered sequence of Steps that defines a manufacturing process.

    A Layer holds its workflows in run order. A Workflow carries the
    operation-level concerns of the layer's parts it runs for: its
    color (used to identify it across the UI), the workpieces it
    operates on (empty means all parts of the layer), its enabled
    flag, and post processors applied to its merged machine-space
    toolpath. It automatically bubbles signals from its child steps.
    """

    DEFAULT_COLOR = COLOR_PALETTE[0]

    def __init__(self, name: str):
        """
        Initializes the Workflow.

        Args:
            name: The user-facing name for the work plan.
        """
        super().__init__(name=name)
        self.per_step_transformer_changed = Signal()
        self.color: str = self.DEFAULT_COLOR
        # The workpieces this workflow runs for, by uid. An empty list
        # means all workpieces of the layer.
        self.workpiece_uids: list[str] = []
        self.enabled: bool = True
        # Workflow-level post processors, applied to the workflow's
        # merged toolpath in machine space (see the machine-transform
        # stage). Serialized transformer dicts, resolved via the
        # transformer registry like step-level transformers.
        self.post_processors_dicts: list[dict[str, Any]] = []
        self.post_processor_changed = Signal()

        # Forward compatibility: store unknown attributes
        self.extra: dict[str, Any] = {}

    def to_dict(self) -> dict:
        """Serializes the workflow and its children to a dictionary."""
        result = {
            "uid": self.uid,
            "type": "workflow",
            "name": self.name,
            "matrix": self.matrix.to_list(),
            "color": self.color,
            "workpiece_uids": list(self.workpiece_uids),
            "enabled": self.enabled,
            "post_processors_dicts": self.post_processors_dicts,
            "children": [child.to_dict() for child in self.children],
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Workflow:
        """Deserializes a dictionary into a Workflow instance."""
        known_keys = {
            "uid",
            "type",
            "name",
            "matrix",
            "color",
            "workpiece_uids",
            "enabled",
            "post_processors_dicts",
            "children",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        workflow = cls(name=data.get("name", "Workflow"))
        workflow.uid = data["uid"]
        workflow.matrix = Matrix.from_list(data["matrix"])
        workflow.color = data.get("color", cls.DEFAULT_COLOR)
        workflow.workpiece_uids = list(data.get("workpiece_uids", []))
        workflow.enabled = data.get("enabled", True)
        workflow.post_processors_dicts = list(
            data.get("post_processors_dicts", [])
        )
        workflow.extra = extra

        steps = [
            Step.from_dict(d)
            for d in data.get("children", [])
            if d.get("type") == "step"
        ]
        workflow.set_children(steps)
        return workflow

    def set_color(self, color: str):
        if self.color == color:
            return
        self.color = color
        self.updated.send(self)

    def set_enabled(self, enabled: bool):
        if self.enabled == enabled:
            return
        self.enabled = enabled
        self.updated.send(self)

    def set_workpiece_uids(self, uids: list[str]):
        if self.workpiece_uids == uids:
            return
        self.workpiece_uids = list(uids)
        self.updated.send(self)

    def selected_workpieces(
        self, workpieces: list[WorkPiece]
    ) -> list[WorkPiece]:
        """
        The given workpieces this workflow runs for, in input order.

        An empty ``workpiece_uids`` filter matches every workpiece of
        the layer; a set filter matches only the workpieces whose uid
        it lists, and silently skips uids that are not present.
        """

        if not self.workpiece_uids:
            return list(workpieces)
        uids = set(self.workpiece_uids)
        return [wp for wp in workpieces if wp.uid in uids]

    def set_post_processors_dicts(
        self, post_processors_dicts: list[dict[str, Any]]
    ):
        if self.post_processors_dicts == post_processors_dicts:
            return
        self.post_processors_dicts = list(post_processors_dicts)
        self.post_processor_changed.send(self)
        self.updated.send(self)

    @property
    def steps(self) -> list[Step]:
        """Returns a list of all child items that are Steps."""
        return [child for child in self.children if isinstance(child, Step)]

    @property
    def layer(self) -> Layer | None:
        """Returns the layer that owns this workflow, if any."""
        from .layer import Layer

        parent = self.parent
        return parent if isinstance(parent, Layer) else None

    @property
    def workpieces(self) -> list[WorkPiece]:
        """The workpieces this workflow runs for, in document order."""
        layer = self.layer
        if layer is None:
            return []
        return self.selected_workpieces(layer.all_workpieces)

    def __iter__(self):
        """Allows iteration over the work steps."""
        return iter(self.steps)

    def _on_per_step_transformer_changed(self, step: Step):
        """
        Handles changes to per-step transformers from a child step and
        bubbles the signal up.
        """
        self.per_step_transformer_changed.send(self)

    def add_child(self, child: T, index: int | None = None) -> T:
        if isinstance(child, Step):
            child.per_step_transformer_changed.connect(
                self._on_per_step_transformer_changed
            )
        super().add_child(child, index)
        return child

    def remove_child(self, child: DocItem):
        if isinstance(child, Step):
            child.per_step_transformer_changed.disconnect(
                self._on_per_step_transformer_changed
            )
        super().remove_child(child)

    def add_children(
        self, children_to_add: Iterable[DocItem], index: int | None = None
    ):
        children = list(children_to_add)
        for child in children:
            if isinstance(child, Step):
                child.per_step_transformer_changed.connect(
                    self._on_per_step_transformer_changed
                )
        super().add_children(children, index)

    def remove_children(self, children_to_remove: Iterable[DocItem]):
        children = list(children_to_remove)
        for child in children:
            if isinstance(child, Step) and child in self.children:
                child.per_step_transformer_changed.disconnect(
                    self._on_per_step_transformer_changed
                )
        super().remove_children(children)

    def set_children(self, new_children: Iterable[DocItem]):
        old_steps = self.steps
        for step in old_steps:
            step.per_step_transformer_changed.disconnect(
                self._on_per_step_transformer_changed
            )

        new_steps = [c for c in new_children if isinstance(c, Step)]
        for step in new_steps:
            step.per_step_transformer_changed.connect(
                self._on_per_step_transformer_changed
            )

        super().set_children(new_children)

    def add_step(self, step: Step):
        """Adds a step to the end of the work plan."""
        self.add_child(step)

    def remove_step(self, step: Step):
        """Removes a step from the work plan."""
        self.remove_child(step)

    def set_steps(self, steps: list[Step]):
        """Replaces the entire list of steps with a new one."""
        self.set_children(steps)

    def has_steps(self) -> bool:
        """Checks if the work plan contains any steps."""
        return len(self.steps) > 0
