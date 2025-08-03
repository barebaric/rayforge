"""
Defines the Workflow class, which holds an ordered sequence of Steps.
"""

from __future__ import annotations
import uuid
import logging
from typing import List, TYPE_CHECKING
from blinker import Signal
from ..config import config
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

    changed = Signal()

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

    def __iter__(self):
        """Allows iteration over the work steps."""
        return iter(self.steps)

    def _on_step_changed(self, step: Step, **kwargs):
        """
        Handles data-changing signals from child steps.

        When a child step's `changed` signal is fired, this method
        catches it and bubbles up the notification by firing the work
        plan's own `changed` signal. This ensures that the parent Layer
        is notified to regenerate operations.
        """
        logger.debug(
            f"Workflow '{self.name}': Notified of model change from step "
            f"'{step.name}'. Firing own changed signal."
        )
        self.changed.send(self, step=step)

    def _connect_step_signals(self, step: Step):
        """Connects the work plan's handlers to a step's signals."""
        logger.debug(f"Connecting signals for step '{step.name}'.")
        step.changed.connect(self._on_step_changed)

        step.ops_generation_starting.connect(
            self.layer._on_step_ops_generation_starting
        )
        step.ops_chunk_available.connect(
            self.layer._on_step_ops_chunk_available
        )
        step.ops_generation_finished.connect(
            self.layer._on_step_ops_generation_finished
        )

    def _disconnect_step_signals(self, step: Step):
        """Disconnects the work plan's handlers from a step's signals."""
        try:
            step.changed.disconnect(self._on_step_changed)
            step.ops_generation_starting.disconnect(
                self.layer._on_step_ops_generation_starting
            )
            step.ops_chunk_available.disconnect(
                self.layer._on_step_ops_chunk_available
            )
            step.ops_generation_finished.disconnect(
                self.layer._on_step_ops_generation_finished
            )
        except TypeError:
            # This can occur if a signal was never connected or was
            # already disconnected, which is safe to ignore.
            pass

    def create_step(self, step_cls, name=None) -> Step:
        """Factory method to create a new step with correct config."""
        return step_cls(
            workflow=self,
            laser=config.machine.heads[0],
            max_cut_speed=config.machine.max_cut_speed,
            max_travel_speed=config.machine.max_travel_speed,
            name=name,
        )

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
        self.changed.send(self)

    def set_steps(self, steps: List[Step]):
        """
        Replaces the entire list of steps with a new one.

        This method efficiently disconnects all signals from the old steps
        and connects signals for all the new ones.

        Args:
            steps: The new list of Step instances.
        """
        for step in self.steps:
            self._disconnect_step_signals(step)
            step.workflow = None

        self.steps = list(steps)

        for step in self.steps:
            step.workflow = self
            self._connect_step_signals(step)

        self.changed.send(self)

    def has_steps(self) -> bool:
        """
        Checks if the work plan contains any steps.

        Returns:
            True if the number of steps is greater than zero, False
            otherwise.
        """
        return len(self.steps) > 0
