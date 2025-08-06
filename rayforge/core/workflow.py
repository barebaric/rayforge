"""
Defines the Workflow class, which holds an ordered sequence of Steps.
"""

from __future__ import annotations
import uuid
import logging
from typing import List, TYPE_CHECKING
from blinker import Signal
from .step import Step

if TYPE_CHECKING:
    from .layer import Layer

logger = logging.getLogger(__name__)


class Workflow:
    """
    An ordered sequence of Steps that defines a manufacturing process.

    Each Layer owns a Workflow. The Workflow holds a list of Step
    objects, which are applied in order to the workpieces in the layer to
    generate machine operations. It listens for changes in its child steps
    and propagates a `changed` signal.
    """

    def __init__(self, layer: "Layer", name: str):
        """
        Initializes the Workflow.

        Args:
            layer: The parent Layer object.
            name: The user-facing name for the work plan.
        """
        self.layer = layer
        self.doc = layer.doc
        self.uid: str = str(uuid.uuid4())
        self.name: str = name
        self.steps: List[Step] = []

        # Ref for static analysis tools to detect class relations.
        self._step_ref_for_pyreverse: Step

        self.changed = Signal()
        self.descendant_added = Signal()
        self.descendant_removed = Signal()
        self.descendant_updated = Signal()

    def __iter__(self):
        """Allows iteration over the work steps."""
        return iter(self.steps)

    def _on_step_changed(self, step: Step):
        """
        Handles data-changing signals from child steps.

        When a child step's `changed` signal is fired, this is interpreted
        as an update to that step. This method fires the `descendant_updated`
        signal with the step as the origin, and the general `changed` signal.
        """
        logger.debug(
            f"Workflow '{self.name}': Notified of model change from step "
            f"'{step.name}'. Firing own signals."
        )
        self.descendant_updated.send(self, origin=step)
        self.changed.send(self)

    def _connect_step_signals(self, step: Step):
        """Connects the work plan's handlers to a step's signals."""
        logger.debug(f"Connecting 'changed' signal for step '{step.name}'.")
        step.changed.connect(self._on_step_changed)

    def _disconnect_step_signals(self, step: Step):
        """Disconnects the work plan's handlers from a step's signals."""
        # This is safe; blinker's disconnect is a no-op if not connected.
        step.changed.disconnect(self._on_step_changed)

    def add_step(self, step: Step):
        """
        Adds a step to the end of the work plan.

        Appends the step, connects its signals, and notifies listeners
        that the work plan has changed.

        Args:
            step: The Step instance to add.
        """
        if step in self.steps:
            return
        if step.workflow and step.workflow is not self:
            step.workflow.remove_step(step)

        step.workflow = self
        self.steps.append(step)
        self._connect_step_signals(step)
        self.descendant_added.send(self, origin=step)
        self.changed.send(self)

    def remove_step(self, step: Step):
        """
        Removes a step from the work plan.

        Disconnects signals from the step, removes it from the list,
        and notifies listeners of the change.

        Args:
            step: The Step instance to remove.
        """
        self._disconnect_step_signals(step)
        self.steps.remove(step)
        step.workflow = None
        self.descendant_removed.send(self, origin=step)
        self.changed.send(self)

    def set_steps(self, steps: List[Step]):
        """
        Replaces the entire list of steps with a new one.

        This method efficiently disconnects all signals from the old steps
        and connects signals for all the new ones.

        Args:
            steps: The new list of Step instances.
        """
        old_set = set(self.steps)
        new_set = set(steps)

        for step in old_set - new_set:
            self._disconnect_step_signals(step)
            step.workflow = None
            self.descendant_removed.send(self, origin=step)

        for step in new_set - old_set:
            if step.workflow and step.workflow is not self:
                step.workflow.remove_step(step)
            step.workflow = self
            self._connect_step_signals(step)
            self.descendant_added.send(self, origin=step)

        # Ensure workflow is set for all steps in the new list
        for step in steps:
            step.workflow = self

        self.steps = list(steps)
        self.changed.send(self)

    def has_steps(self) -> bool:
        """
        Checks if the work plan contains any steps.

        Returns:
            True if the number of steps is greater than zero, False
            otherwise.
        """
        return len(self.steps) > 0
