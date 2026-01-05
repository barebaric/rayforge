import logging
from typing import TYPE_CHECKING, Optional, cast
from gi.repository import GLib
from ....core.workpiece import WorkPiece
from ....core.group import Group
from ...canvas import ShrinkWrapGroup
from .workpiece import WorkPieceElement

if TYPE_CHECKING:
    from ..surface import WorkSurface
    from ...canvas import CanvasElement


logger = logging.getLogger(__name__)


class GroupElement(ShrinkWrapGroup):
    """
    A CanvasElement that represents a Group data model.
    """

    def __init__(self, group: "Group", **kwargs):
        # The element is "passive" during its entire construction and the
        # synchronous execution of the command that creates it.
        self._is_passive = True
        super().__init__(data=group, pixel_perfect_hit=True, **kwargs)
        self.data.updated.connect(self.sync_with_model)
        self.data.transform_changed.connect(self._on_transform_changed)
        self.data.descendant_added.connect(self.sync_with_model)
        self.data.descendant_removed.connect(self.sync_with_model)

        # Set the initial transform from the model. This is critical.
        self._on_transform_changed(self.data)

        # Build the child view hierarchy.
        self.sync_with_model()

        # Schedule the group to become "active" and
        # perform its first shrink-wrap calculation in the next idle cycle.
        # This guarantees that the CreateGroupCommand has completely finished
        # setting up the model state before the view tries to react to it.
        GLib.idle_add(self._activate_and_update)

    def _activate_and_update(self) -> bool:
        """Callback to activate the element and run its first update."""
        self._is_passive = False
        self.update_bounds()
        if self.canvas:
            self.canvas.queue_draw()
        return GLib.SOURCE_REMOVE  # Run only once

    def on_child_transform_changed(self, child: "CanvasElement"):
        """Override to prevent updates while the group is passive."""
        if self._is_passive:
            return  # Ignore all child updates during initialization

        # After activation, use the standard ShrinkWrapGroup behavior
        super().on_child_transform_changed(child)

    def remove(self):
        """Disconnects signals before removing the element."""
        self.data.updated.disconnect(self.sync_with_model)
        self.data.transform_changed.disconnect(self._on_transform_changed)
        self.data.descendant_added.disconnect(self.sync_with_model)
        self.data.descendant_removed.disconnect(self.sync_with_model)
        super().remove()

    def set_ops_visibility(self, step_uid: str, visible: bool):
        """
        Propagates the ops visibility setting to all child elements.
        """
        for child in self.children:
            assert isinstance(child, (WorkPieceElement, GroupElement))
            child.set_ops_visibility(step_uid, visible)

    def get_elem_hit(
        self, world_x: float, world_y: float, selectable: bool = False
    ) -> Optional["CanvasElement"]:
        """
        Overrides the default hit-test to enforce group selection behavior.

        If any element within this group's hierarchy (a child, a grandchild,
        or the group's own body) is visually under the cursor, this method
        intercepts the result and returns the group itself, provided the group
        is selectable. This makes the entire group act as a single unit for
        both click and frame selection.
        """
        # If the caller requires a selectable element and this group isn't,
        # quit.
        if selectable and not self.selectable:
            return None

        # Check for a visual hit within the group's hierarchy, ignoring the
        # individual `selectable` flags of children. This is the key to making
        # the group an atomic unit for click-selection.
        hit_candidate = super().get_elem_hit(
            world_x, world_y, selectable=False
        )

        # If a visual component was hit, the hit is on the group itself.
        return self if hit_candidate else None

    def _on_transform_changed(self, group: Group):
        """
        Handles transform changes from the model by applying the model's
        local matrix to this canvas element's transform. (MODEL -> VIEW)
        """
        if self.transform != group.matrix:
            self.set_transform(group.matrix)

    def push_transform_to_model(self):
        """
        Updates the data model with the current transformation matrix from the
        view. Called by the WorkSurface at the end of an interactive operation.
        (VIEW -> MODEL)
        """
        if self.data.matrix != self.transform:
            logger.debug(
                "[GroupElem] VIEW->MODEL: Pushing transform for"
                f" '{self.data.name}'"
            )
            self.data.matrix = self.transform.copy()

    def sync_with_model(self, *args, **kwargs):
        """
        Reconciles child elements (WorkPieceElement, GroupElement) with the
        state of the Group model.
        """
        if not self.data or not self.canvas:
            return

        work_surface = cast("WorkSurface", self.canvas)
        model_children = set(self.data.children)
        current_elements = self.children[:]
        current_element_data = {elem.data for elem in current_elements}

        # Remove elements for items no longer in the group
        for elem in current_elements:
            if elem.data not in model_children:
                elem.remove()

        # Add elements for new items in the group
        items_to_add = model_children - current_element_data
        for item_data in items_to_add:
            child_elem = None
            if isinstance(item_data, WorkPiece):
                child_elem = WorkPieceElement(
                    workpiece=item_data,
                    pipeline=work_surface.editor.pipeline,
                    canvas=self.canvas,
                    selectable=False,  # Children are not selectable
                )
            elif isinstance(item_data, Group):
                child_elem = GroupElement(
                    group=item_data,
                    canvas=self.canvas,
                    selectable=False,  # Children are not selectable
                )

            if child_elem:
                self.add(child_elem)

        # Do NOT call update_bounds here. It's handled by the activation
        # callback or the on_child_transform_changed logic.
        if self.canvas:
            self.canvas.queue_draw()
