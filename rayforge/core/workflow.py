"""
Defines the Workflow class, which holds an ordered sequence of Steps.
"""

from __future__ import annotations
import logging
from typing import List, Optional, TypeVar, Iterable, Dict
from blinker import Signal
from .item import DocItem
from .step import Step


logger = logging.getLogger(__name__)

# For generic type hinting in add_child
T = TypeVar("T", bound="DocItem")


class Workflow(DocItem):
    """
    An ordered sequence of Steps that defines a manufacturing process.

    Each Layer owns a Workflow. The Workflow holds a list of Step
    objects, which are applied in order to the workpieces in the layer to
    generate machine operations. It automatically bubbles signals from its
    child steps.
    """

    def __init__(self, name: str):
        """
        Initializes the Workflow.

        Args:
            name: The user-facing name for the work plan.
        """
        super().__init__(name=name)
        self.post_step_transformer_changed = Signal()

    def to_dict(self) -> Dict:
        """Serializes the workflow and its children to a dictionary."""
        return {
            "uid": self.uid,
            "type": "workflow",
            "name": self.name,
            "matrix": self.matrix.to_list(),
            "children": [child.to_dict() for child in self.children],
        }

    @property
    def steps(self) -> List[Step]:
        """Returns a list of all child items that are Steps."""
        return [child for child in self.children if isinstance(child, Step)]

    def __iter__(self):
        """Allows iteration over the work steps."""
        return iter(self.steps)

    def _on_post_step_transformer_changed(self, step: Step):
        """
        Handles changes to post-step transformers from a child step and
        bubbles the signal up.
        """
        self.post_step_transformer_changed.send(self)

    def add_child(self, child: T, index: Optional[int] = None) -> T:
        if isinstance(child, Step):
            child.post_step_transformer_changed.connect(
                self._on_post_step_transformer_changed
            )
        super().add_child(child, index)
        return child

    def remove_child(self, child: DocItem):
        if isinstance(child, Step):
            child.post_step_transformer_changed.disconnect(
                self._on_post_step_transformer_changed
            )
        super().remove_child(child)

    def set_children(self, new_children: Iterable[DocItem]):
        old_steps = self.steps
        for step in old_steps:
            step.post_step_transformer_changed.disconnect(
                self._on_post_step_transformer_changed
            )

        new_steps = [c for c in new_children if isinstance(c, Step)]
        for step in new_steps:
            step.post_step_transformer_changed.connect(
                self._on_post_step_transformer_changed
            )

        super().set_children(new_children)

    def add_step(self, step: Step):
        """Adds a step to the end of the work plan."""
        self.add_child(step)

    def remove_step(self, step: Step):
        """Removes a step from the work plan."""
        self.remove_child(step)

    def set_steps(self, steps: List[Step]):
        """Replaces the entire list of steps with a new one."""
        self.set_children(steps)

    def has_steps(self) -> bool:
        """Checks if the work plan contains any steps."""
        return len(self.steps) > 0
