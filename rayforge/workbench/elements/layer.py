import logging
from typing import TYPE_CHECKING, cast, Optional

from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from ...core.group import Group
from ..canvas.element import CanvasElement
from .workpiece import WorkPieceView
from .step import StepElement
from .group import GroupElement

if TYPE_CHECKING:
    from ...core.layer import Layer


logger = logging.getLogger(__name__)


def _z_order_sort_key(element: CanvasElement):
    """Sort key to ensure StepElements are drawn after visual elements."""
    if isinstance(element, (WorkPieceView, GroupElement)):
        return 0  # Draw workpieces and groups first (at the bottom)
    if isinstance(element, StepElement):
        # StepElements are invisible managers, but we keep them in the
        # sort order for consistency.
        return 1
    return 2


class LayerElement(CanvasElement):
    """
    A non-selectable container that corresponds to a Layer model.
    It creates and manages child elements for WorkPieces, Groups, and Steps.
    """

    def __init__(self, layer: "Layer", **kwargs):
        super().__init__(
            x=0,
            y=0,
            width=0,
            height=0,
            selectable=False,
            background=(0, 0, 0, 0),
            clip=False,
            data=layer,
            **kwargs,
        )
        self.data: Layer = layer
        self.data.updated.connect(self.sync_with_model)
        self.data.descendant_added.connect(self.sync_with_model)
        self.data.descendant_removed.connect(self.sync_with_model)
        self.sync_with_model(origin=None)

    def remove(self):
        """Disconnects signals before removing the element."""
        self.data.updated.disconnect(self.sync_with_model)
        self.data.descendant_added.disconnect(self.sync_with_model)
        self.data.descendant_removed.disconnect(self.sync_with_model)
        super().remove()

    def set_size(self, width: float, height: float):
        """Sets the size and propagates it to child StepElements."""
        if self.width == width and self.height == height:
            return
        super().set_size(width, height)

        # StepElements are invisible but might need size for future features.
        for elem in self.children:
            if isinstance(elem, StepElement):
                elem.set_size(width, height)

    def sort_children_by_z_order(self):
        """Sorts child elements to maintain correct drawing order."""
        self.children.sort(key=_z_order_sort_key)

    def sync_with_model(
        self, *args, origin: Optional[DocItem] = None, **kwargs
    ):
        """
        Reconciles all child elements with the state of the Layer model.
        """
        if not self.data or not self.canvas:
            return

        logger.debug(
            f"LayerElement for '{self.data.name}': sync_with_model is"
            f" executing, called by {origin}."
        )
        self.set_visible(self.data.visible)
        from ..surface import WorkSurface

        work_surface = cast(WorkSurface, self.canvas)

        # Reconcile Visual Elements (WorkPieces and Groups)
        model_items = {
            c for c in self.data.children if isinstance(c, (WorkPiece, Group))
        }
        current_visual_elements = [
            elem
            for elem in self.children
            if isinstance(elem, (WorkPieceView, GroupElement))
        ]

        # Remove elements for items no longer in the layer
        for elem in current_visual_elements[:]:
            if elem.data not in model_items:
                logger.debug(f"Removing visual element: {elem}")
                elem.remove()

        # Add elements for new items in the layer.
        current_item_data = {elem.data for elem in self.children}
        items_to_add = model_items - current_item_data
        for item_data in items_to_add:
            new_elem = None
            if isinstance(item_data, WorkPiece):
                new_elem = WorkPieceView(
                    workpiece=item_data,
                    ops_generator=work_surface.ops_generator,
                    canvas=self.canvas,
                    selectable=self.data.visible,
                )
                new_elem.set_base_image_visible(
                    work_surface.are_workpieces_visible()
                )
            elif isinstance(item_data, Group):
                new_elem = GroupElement(
                    group=item_data,
                    canvas=self.canvas,
                    selectable=self.data.visible,
                )
            if new_elem:
                self.add(new_elem)

        # Reconcile StepElements (Lifecycle Managers)
        current_step_elements = [
            elem for elem in self.children if isinstance(elem, StepElement)
        ]
        workpiece_views = [
            elem for elem in self.children if isinstance(elem, WorkPieceView)
        ]
        model_steps = set(self.data.workflow.steps)

        # Remove StepElements for steps that are no longer in the model
        for elem in current_step_elements:
            if elem.data not in model_steps:
                removed_step_uid = elem.data.uid
                logger.debug(
                    "LayerElement detected removal of step "
                    f"'{elem.data.name}'. "
                    f"Cleaning up visuals for UID {removed_step_uid}."
                )
                # Instruct all workpiece views in this layer to clear the
                # surface
                for wp_view in workpiece_views:
                    wp_view.clear_ops_surface(removed_step_uid)
                elem.remove()

        # Add StepElements for new steps.
        current_step_data = {
            elem.data
            for elem in self.children
            if isinstance(elem, StepElement)
        }
        steps_to_add = model_steps - current_step_data
        for step_data in steps_to_add:
            step_elem = StepElement(
                step=step_data,
                ops_generator=work_surface.ops_generator,
                canvas=self.canvas,
            )
            self.add(step_elem)

        # After all children are added/removed, we ensure every StepElement
        # broadcasts its current visibility to every WorkPieceView. This
        # guarantees that newly created elements get the correct initial state.
        for elem in self.children:
            if isinstance(elem, StepElement):
                elem._update_sibling_ops_visibility()

        self.sort_children_by_z_order()
        self.canvas.queue_draw()
