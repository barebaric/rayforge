"""
Defines the Layer class, a central component for organizing and processing
workpieces within a document.
"""

from __future__ import annotations
import uuid
import logging
from typing import List, TYPE_CHECKING, Tuple
from blinker import Signal

from ..core.step import Step
from ..core.workflow import Workflow

if TYPE_CHECKING:
    from .workpiece import WorkPiece
    from .doc import Doc

logger = logging.getLogger(__name__)


class Layer:
    """
    Represents a group of workpieces processed by a single workflow.

    A Layer acts as a container for `WorkPiece` objects and owns a
    `Workflow`. It is responsible for triggering, managing, and caching the
    generation of machine operations (`Ops`) for each workpiece based on the
    steps in its workflow.
    """

    def __init__(self, doc: "Doc", name: str):
        """Initializes a Layer instance.

        Args:
            doc: The parent document object.
            name: The user-facing name of the layer.
        """
        self.uid: str = str(uuid.uuid4())
        self.doc: "Doc" = doc
        self.name: str = name
        self.workpieces: List[WorkPiece] = []
        # Reference for static analysis tools to detect class relations.
        self._workpiece_ref_for_pyreverse: WorkPiece

        self.workflow: Workflow = Workflow(self, f"{name} Workflow")
        # Reference for static analysis tools to detect class relations.
        self._workflow_ref_for_pyreverse: Workflow

        self.visible: bool = True

        # Signals for notifying other parts of the application of changes.
        self.changed = Signal()
        self.post_step_transformer_changed = Signal()
        self.descendant_added = Signal()
        self.descendant_removed = Signal()
        self.descendant_updated = Signal()
        self.descendant_transform_changed = Signal()

        # Connect to signals from child objects.
        self.workflow.changed.connect(self._on_workflow_changed)
        self.workflow.post_step_transformer_changed.connect(
            self._on_workflow_post_transformer_changed
        )
        self.workflow.descendant_added.connect(self._on_descendant_added)
        self.workflow.descendant_removed.connect(self._on_descendant_removed)
        self.workflow.descendant_updated.connect(self._on_descendant_updated)

    @property
    def active(self) -> bool:
        """
        Returns True if this layer is the currently active layer in the
        document.
        """
        return self.doc.active_layer is self

    def _on_workflow_changed(self, sender):
        """
        Handles the 'changed' signal from the Workflow and bubbles it up.
        """
        logger.debug(
            f"Layer '{self.name}': Noticed workflow change, bubbling up."
        )
        self.changed.send(self)

    def _on_workflow_post_transformer_changed(self, sender):
        """
        Bubbles up the post_transformer_changed signal from the workflow.
        """
        self.post_step_transformer_changed.send(self)

    def _on_descendant_added(self, sender, *, origin):
        """Bubbles up the descendant_added signal."""
        self.descendant_added.send(self, origin=origin)

    def _on_descendant_removed(self, sender, *, origin):
        """Bubbles up the descendant_removed signal."""
        self.descendant_removed.send(self, origin=origin)

    def _on_descendant_updated(self, sender, *, origin):
        """Bubbles up the descendant_updated signal."""
        self.descendant_updated.send(self, origin=origin)

    def _on_workpiece_changed(self, workpiece: WorkPiece):
        """
        Handles data-changing signals from a workpiece (e.g., size change)
        that require ops regeneration.
        """
        logger.debug(
            f"Layer '{self.name}': Noticed model change for "
            f"'{workpiece.name}', bubbling up."
        )
        self.descendant_updated.send(self, origin=workpiece)
        self.changed.send(self)

    def _on_workpiece_transform_changed(self, workpiece: WorkPiece):
        """
        Handles transform-only changes from a workpiece. This bubbles up a
        specific signal that the Doc can use to mark itself as dirty, but
        that the LayerElement can ignore to prevent slow UI updates.
        """
        self.descendant_transform_changed.send(self, origin=workpiece)

    def set_name(self, name: str):
        """Sets the name of the layer.

        Args:
            name: The new name for the layer.
        """
        if self.name == name:
            return
        self.name = name
        self.workflow.name = f"{name} Workflow"
        self.changed.send(self)

    def set_visible(self, visible: bool):
        """Sets the visibility of the layer.

        Args:
            visible: The new visibility state.
        """
        if self.visible == visible:
            return
        self.visible = visible
        self.changed.send(self)

    def add_workpiece(self, workpiece: "WorkPiece"):
        """Adds a single workpiece to the layer.

        Args:
            workpiece: The workpiece to add.
        """
        if workpiece not in self.workpieces:
            workpiece.parent = self
            self.workpieces.append(workpiece)
            workpiece.changed.connect(self._on_workpiece_changed)
            workpiece.transform_changed.connect(
                self._on_workpiece_transform_changed
            )
            self.descendant_added.send(self, origin=workpiece)
            self.changed.send(self)

    def remove_workpiece(self, workpiece: "WorkPiece"):
        """Removes a single workpiece from the layer.

        Args:
            workpiece: The workpiece to remove.
        """
        if workpiece in self.workpieces:
            workpiece.parent = None
            workpiece.changed.disconnect(self._on_workpiece_changed)
            workpiece.transform_changed.disconnect(
                self._on_workpiece_transform_changed
            )
            self.workpieces.remove(workpiece)
            self.descendant_removed.send(self, origin=workpiece)
            self.changed.send(self)

    def set_workpieces(self, workpieces: List["WorkPiece"]):
        """
        Sets the layer's workpieces to a new list.

        Args:
            workpieces: A list of WorkPiece objects to associate with
                this layer.
        """
        old_set = set(self.workpieces)
        new_set = set(workpieces)

        # Disconnect and clean up workpieces that are being removed.
        for wp in old_set - new_set:
            wp.parent = None
            wp.changed.disconnect(self._on_workpiece_changed)
            wp.transform_changed.disconnect(
                self._on_workpiece_transform_changed
            )
            self.descendant_removed.send(self, origin=wp)

        # Connect new workpieces.
        for wp in new_set - old_set:
            wp.parent = self
            wp.changed.connect(self._on_workpiece_changed)
            wp.transform_changed.connect(
                self._on_workpiece_transform_changed
            )
            self.descendant_added.send(self, origin=wp)

        self.workpieces = list(workpieces)
        self.changed.send(self)

    def get_renderable_items(self) -> List[Tuple[Step, WorkPiece]]:
        """
        Gets a list of all visible step/workpiece pairs for rendering.

        Returns:
            A list of (Step, WorkPiece) tuples that are currently
            visible and have valid geometry for rendering.
        """
        if not self.visible:
            return []
        items = []
        for workpiece in self.workpieces:
            if any(s <= 0 for s in workpiece.size):
                continue
            for step in self.workflow.steps:
                if step.visible:
                    items.append((step, workpiece))
        return items
