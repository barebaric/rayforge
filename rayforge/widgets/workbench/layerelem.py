import logging
from typing import Optional, TYPE_CHECKING, cast
from ..canvas.element import CanvasElement
from .workpieceelem import WorkPieceElement
from .stepelem import StepElement

if TYPE_CHECKING:
    from ...models.layer import Layer


logger = logging.getLogger(__name__)


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
                workpiece=wp_data, canvas=self.canvas, selectable=is_selectable
            )
            self.add(wp_elem)
            # Position and size the new element based on model data
            wp_elem.allocate()

        # --- Reconcile StepElements ---
        # Now add/remove the StepElements themselves.
        current_ws_elements = self.find_by_type(StepElement)
        model_steps = set(self.data.workflow.steps)
        current_ws_data = {elem.data for elem in current_ws_elements}

        # Remove elements for steps that are no longer in the layer's workplan
        for elem in current_ws_elements:
            if elem.data not in model_steps:
                elem.remove()

        # Add elements for new steps in the layer's workplan
        # Use local import to break circular dependency
        #  (surface -> layerelem -> surface)
        from .surface import WorkSurface

        work_surface = cast(WorkSurface, self.canvas)
        show_travel = (
            work_surface._show_travel_moves if work_surface else False
        )

        wss_to_add = model_steps - current_ws_data
        for ws_data in wss_to_add:
            ws_elem = StepElement(
                step=ws_data,
                x=0,
                y=0,
                width=self.width,
                height=self.height,
                show_travel_moves=show_travel,
                canvas=self.canvas,
                parent=self,  # Explicitly set parent
            )
            self.add(ws_elem)

        self.canvas.queue_draw()

    def get_elem_hit(
        self, x: float, y: float, selectable: bool = False
    ) -> Optional[CanvasElement]:
        if not self.visible:
            return None
        return super().get_elem_hit(x, y, selectable)
