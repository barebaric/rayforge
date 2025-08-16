import logging
from typing import TYPE_CHECKING, cast, Optional, List, Dict, Tuple
from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from ...core.group import Group
from ..canvas.element import CanvasElement
from .workpiece import WorkPieceElement
from .step import StepElement
from .ops import WorkPieceOpsElement
from .group import GroupElement

if TYPE_CHECKING:
    from ...core.layer import Layer


logger = logging.getLogger(__name__)


def _z_order_sort_key(element: CanvasElement):
    """
    Sort key to ensure StepElements are drawn on top of WorkPieceElements.
    """
    if isinstance(element, (WorkPieceElement, GroupElement)):
        return 0  # Draw workpieces and groups first (at the bottom)
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
        # A cache to track world scales to detect changes from parent
        # transforms
        self._wp_scale_cache: Dict[str, Tuple[float, float]] = {}
        self.data.updated.connect(self.sync_with_model)
        self.data.descendant_added.connect(self.sync_with_model)
        self.data.descendant_removed.connect(self.sync_with_model)
        self.data.descendant_transform_changed.connect(
            self._on_descendant_transform_changed
        )
        self.sync_with_model(origin=None)

    def remove(self):
        """Disconnects signals before removing the element."""
        self.data.updated.disconnect(self.sync_with_model)
        self.data.descendant_added.disconnect(self.sync_with_model)
        self.data.descendant_removed.disconnect(self.sync_with_model)
        self.data.descendant_transform_changed.disconnect(
            self._on_descendant_transform_changed
        )
        super().remove()

    def _on_descendant_transform_changed(
        self, sender: DocItem, *, origin: DocItem, **kwargs
    ):
        """
        Handles bubbled transform changes from any descendant (e.g., a Group).
        This ensures that WorkPieceOpsElements are updated when their effective
        world transform changes, and that ops are regenerated if the scale
        changed.
        """
        affected_wps = [origin]
        affected_wps.extend(origin.get_descendants(WorkPiece))
        if not affected_wps:
            return

        for wp in affected_wps:
            # Check for world scale change to trigger ops regeneration
            new_scale = wp.get_world_transform().get_abs_scale()
            old_scale = self._wp_scale_cache.get(wp.uid)

            # If scale changed, fire the model's updated signal to trigger
            # a regeneration of the ops content.
            if old_scale and (
                abs(old_scale[0] - new_scale[0]) > 1e-9
                or abs(old_scale[1] - new_scale[1]) > 1e-9
            ):
                logger.debug(
                    f"World scale for {wp.name} changed due to ancestor. "
                    "Regenerating ops."
                )
                wp.updated.send(wp)

            # Always update the cache with the latest scale for the next event
            self._wp_scale_cache[wp.uid] = new_scale

            # Update the view elements (the ops containers) to match the new
            # transform.
            for step_elem in self.find_by_type(StepElement):
                ops_elem = cast(
                    WorkPieceOpsElement, step_elem.find_by_data(wp)
                )
                if ops_elem:
                    ops_elem._on_workpiece_transform_changed(wp)

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

    def sync_with_model(
        self, *args, origin: Optional[DocItem] = None, **kwargs
    ):
        """
        Updates the element's properties and reconciles all child elements
        (WorkPieceElement, GroupElement, StepElement) with the state of the
        Layer model.
        """
        if not self.data or not self.canvas:
            return

        logger.debug(
            f"LayerElement for '{self.data.name}': sync_with_model is"
            f" executing, called by {origin}."
        )
        self.set_visible(self.data.visible)
        is_selectable = self.data.visible

        # Use local import to break circular dependency and get canvas type
        from ..surface import WorkSurface

        work_surface = cast(WorkSurface, self.canvas)

        # --- Reconcile Visual Elements (WorkPieces and Groups) ---
        model_items = {
            c for c in self.data.children if isinstance(c, (WorkPiece, Group))
        }

        current_visual_elements: List[CanvasElement] = [
            elem
            for elem in self.children
            if isinstance(elem, (WorkPieceElement, GroupElement))
        ]

        # Remove elements for items no longer in the layer
        for elem in current_visual_elements[:]:  # Iterate over a copy
            if elem.data not in model_items:
                logger.debug(f"Removing visual element: {elem}")
                elem.remove()
            else:
                elem.selectable = is_selectable

        # Add elements for new items in the layer.
        current_item_data = {
            elem.data
            for elem in self.children
            if isinstance(elem, (WorkPieceElement, GroupElement))
        }

        items_to_add = model_items - current_item_data
        for item_data in items_to_add:
            new_elem = None
            if isinstance(item_data, WorkPiece):
                new_elem = WorkPieceElement(
                    workpiece=item_data,
                    canvas=self.canvas,
                    selectable=is_selectable,
                    visible=work_surface._workpieces_visible,
                )
            elif isinstance(item_data, Group):
                new_elem = GroupElement(
                    group=item_data,
                    canvas=self.canvas,
                    selectable=is_selectable,
                )

            if new_elem:
                self.add(new_elem)
                if isinstance(new_elem, WorkPieceElement):
                    new_elem.allocate()

        # --- Reconcile StepElements ---
        current_step_elements = [
            elem for elem in self.children if isinstance(elem, StepElement)
        ]
        model_steps = set(self.data.workflow.steps)

        # Remove StepElements for steps that are no longer in the model
        for elem in current_step_elements:
            if elem.data not in model_steps:
                logger.debug(f"Removing step element: {elem}")
                elem.remove()

        # Add StepElements for new steps.
        current_step_data = {
            elem.data
            for elem in self.children
            if isinstance(elem, StepElement)
        }
        show_travel = (
            work_surface._show_travel_moves if work_surface else False
        )
        ops_generator = work_surface.ops_generator

        steps_to_add = model_steps - current_step_data
        for step_data in steps_to_add:
            step_elem = StepElement(
                step=step_data,
                ops_generator=ops_generator,
                x=0,
                y=0,
                width=self.width,
                height=self.height,
                show_travel_moves=show_travel,
                canvas=self.canvas,
            )
            self.add(step_elem)

        # Now that StepElements are correct, tell them to sync their children.
        for elem in self.children:
            if isinstance(elem, StepElement):
                elem._on_step_model_changed(elem.data)

        # After all reconciliation, prime/update the scale cache.
        self._wp_scale_cache = {
            wp.uid: wp.get_world_transform().get_abs_scale()
            for wp in self.data.all_workpieces
        }

        self.sort_children_by_z_order()
        self.canvas.queue_draw()
