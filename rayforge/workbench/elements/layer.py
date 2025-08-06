import logging
from typing import Optional, TYPE_CHECKING, cast
from ..canvas.element import CanvasElement
from .workpiece import WorkPieceElement
from .step import StepElement

if TYPE_CHECKING:
    from ...core.layer import Layer


logger = logging.getLogger(__name__)


def _z_order_sort_key(element: CanvasElement):
    """
    Sort key to ensure StepElements are drawn on top of WorkPieceElements.
    """
    if isinstance(element, WorkPieceElement):
        return 0  # Draw workpieces first (at the bottom)
    if isinstance(element, StepElement):
        return 1  # Draw step ops on top of workpieces
    return 2  # Other elements on top


class LayerElement(CanvasElement):
    """
    A non-selectable, non-visible container element in the canvas that
    directly corresponds to a Layer model. Its state is managed by the
    WorkSurface.
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
        self.data.changed.connect(self.sync_with_model)

    def set_size(self, width: float, height: float):
        """Sets the size and propagates it to child StepElements."""
        if self.width == width and self.height == height:
            return
        super().set_size(width, height)
        for elem in self.children:
            if isinstance(elem, StepElement):
                elem.set_size(width, height)

    def sort_children_by_z_order(self):
        """Sorts child elements to maintain correct drawing order."""
        self.children.sort(key=_z_order_sort_key)

    def sync_with_model(self, *args, **kwargs):
        """
        Updates the element's properties and reconciles all child elements
        (WorkPieceElement, StepElement) with the state of the Layer model.
        """
        if not self.data or not self.canvas:
            return

        logger.debug(
            f"LayerElement for '{self.data.name}': sync_with_model is"
            " executing."
        )
        self.set_visible(self.data.visible)
        is_selectable = self.data.visible

        # Use local import to break circular dependency and get canvas type
        from ..surface import WorkSurface

        work_surface = cast(WorkSurface, self.canvas)

        # --- Reconcile WorkPieceElements ---
        model_workpieces = set(self.data.workpieces)
        current_wp_elements = {
            child
            for child in self.children
            if isinstance(child, WorkPieceElement)
        }
        current_wp_data = {elem.data for elem in current_wp_elements}

        # Remove elements for workpieces that are no longer in the layer
        # and update selectability on the ones that remain.
        for elem in current_wp_elements:
            if elem.data not in model_workpieces:
                elem.remove()
            else:
                elem.selectable = is_selectable

        # Add elements for new workpieces in the layer
        wps_to_add = model_workpieces - current_wp_data
        for wp_data in wps_to_add:
            wp_elem = WorkPieceElement(
                workpiece=wp_data,
                canvas=self.canvas,
                selectable=is_selectable,
                visible=work_surface._workpieces_visible,
            )
            self.add(wp_elem)
            # Position and size the new element based on model data
            wp_elem.allocate()

        # --- Reconcile StepElements ---
        # Now add/remove the StepElements themselves.
        current_ws_elements = self.find_by_type(StepElement)
        model_steps = set(self.data.workflow.steps)

        # Create a list of elements to remove first.
        elements_to_remove = []
        for elem in current_ws_elements:
            if elem.data not in model_steps:
                elements_to_remove.append(elem)

        # Now, iterate over the list to perform the removal.
        # This avoids modifying the list we are iterating over.
        for elem in elements_to_remove:
            elem.remove()

        # Add elements for new steps in the layer's workplan
        current_ws_data = {
            elem.data for elem in self.find_by_type(StepElement)
        }
        show_travel = (
            work_surface._show_travel_moves if work_surface else False
        )
        ops_generator = work_surface.ops_generator

        wss_to_add = model_steps - current_ws_data
        for ws_data in wss_to_add:
            ws_elem = StepElement(
                step=ws_data,
                ops_generator=ops_generator,
                x=0,
                y=0,
                width=self.width,
                height=self.height,
                show_travel_moves=show_travel,
                canvas=self.canvas,
                parent=self,  # Explicitly set parent
            )
            self.add(ws_elem)

        # Now that the set of StepElements is correct, tell all of them to
        # reconcile their children against the (possibly changed) workpiece
        # list.
        for elem in self.find_by_type(StepElement):
            elem = cast(StepElement, elem)
            elem._on_step_model_changed(elem.data)

        # Sort the children to ensure that all StepElements are drawn on
        # top of all WorkPieceElements.
        self.sort_children_by_z_order()

        self.canvas.queue_draw()

    def get_elem_hit(
        self, x: float, y: float, selectable: bool = False
    ) -> Optional[CanvasElement]:
        if not self.visible:
            return None
        return super().get_elem_hit(x, y, selectable)
