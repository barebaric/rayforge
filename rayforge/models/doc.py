import cairo
import asyncio
import logging
from typing import List, Optional
from blinker import Signal
from pathlib import Path
from ..config import config
from ..undo import HistoryManager
from ..tasker.manager import CancelledError
from ..tasker.context import ExecutionContext
from .workpiece import WorkPiece
from .step import Contour
from .layer import Layer
from .ops import Ops


logger = logging.getLogger(__name__)


class Doc:
    """
    Represents a loaded Rayforge document.
    """

    def __init__(self):
        self.history_manager = HistoryManager()
        self.changed = Signal()
        self.active_layer_changed = Signal()

        self.layers: List[Layer] = []
        self._layer_ref_for_pyreverse: Layer
        self._active_layer_index: int = 0

        layer = Layer(self, _("Layer 1"))
        self.add_layer(layer)

        # Create a default Contour step and add it to the first layer's
        # workflow.
        workflow = layer.workflow
        default_step = workflow.create_step(Contour)
        workflow.add_step(default_step)

    def __iter__(self):
        """Iterates through all workpieces in all layers."""
        return (wp for layer in self.layers for wp in layer.workpieces)

    @property
    def workpieces(self) -> List[WorkPiece]:
        """Returns a list of all workpieces from all layers."""
        return list(self)

    def add_workpiece(self, workpiece: WorkPiece):
        """Adds a workpiece to the currently active layer."""
        self.active_layer.add_workpiece(workpiece)

    def remove_workpiece(self, workpiece: WorkPiece):
        """Removes a workpiece from the layer that owns it."""
        if workpiece.layer and workpiece.layer in self.layers:
            workpiece.layer.remove_workpiece(workpiece)

    @property
    def active_layer(self) -> Layer:
        """Returns the currently active layer."""
        return self.layers[self._active_layer_index]

    def set_active_layer(self, layer: Layer):
        """Sets the active layer by instance."""
        try:
            new_index = self.layers.index(layer)
            if self._active_layer_index != new_index:
                self._active_layer_index = new_index
                self.changed.send(self)
                self.active_layer_changed.send(self)
        except ValueError:
            logger.warning("Attempted to set a non-existent layer as active.")

    def _on_model_changed(self, sender, **kwargs):
        """A single handler for changes in layers."""
        # Just bubble up the notification. The document state has changed.
        self.changed.send(self)

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        layer.changed.connect(self._on_model_changed)
        self.changed.send(self)

    def remove_layer(self, layer: Layer):
        # Prevent removing the last layer.
        if layer not in self.layers or len(self.layers) <= 1:
            return
        try:
            layer.changed.disconnect(self._on_model_changed)
        except TypeError:
            pass
        self.layers.remove(layer)

        # Ensure active_layer_index remains valid
        if self._active_layer_index >= len(self.layers):
            self._active_layer_index = len(self.layers) - 1
            self.active_layer_changed.send(self)

        self.changed.send(self)

    def set_layers(self, layers: List[Layer]):
        # A document must always have at least one layer.
        if not layers:
            raise ValueError("Workpiece layer list cannot be empty.")

        # Preserve the active layer if it still exists in the new list
        current_active = self.active_layer
        old_active_index = self._active_layer_index
        try:
            new_active_index = layers.index(current_active)
        except ValueError:
            new_active_index = 0  # Default to first layer

        for layer in self.layers:
            try:
                layer.changed.disconnect(self._on_model_changed)
            except TypeError:
                pass
        self.layers = list(layers)
        for layer in self.layers:
            layer.changed.connect(self._on_model_changed)

        self._active_layer_index = new_active_index
        self.changed.send(self)
        if old_active_index != self._active_layer_index:
            self.active_layer_changed.send(self)

    def has_workpiece(self):
        return bool(self.workpieces)

    def has_result(self):
        # A result is possible if there's a workpiece and at least one
        # workflow (in any layer) has steps.
        return self.has_workpiece() and any(
            layer.workflow.has_steps() for layer in self.layers
        )

    async def generate_job_ops(
        self, context: Optional[ExecutionContext] = None
    ) -> Ops:
        """
        Executes all steps in all assigned workflows and returns the
        final, combined Ops for the entire job.
        """
        final_ops = Ops()
        machine_width, machine_height = config.machine.dimensions
        clip_rect = 0, 0, machine_width, machine_height

        work_items = []
        for layer in self.layers:
            work_items.extend(layer.get_renderable_items())

        if not work_items:
            return final_ops

        total_items = len(work_items)
        for i, (step, workpiece) in enumerate(work_items):
            if context:
                if context.is_cancelled():
                    raise CancelledError("Operation cancelled")
                context.set_progress(i / total_items)
                context.set_message(
                    _("Processing '{workpiece}' in '{step}'").format(
                        workpiece=workpiece.name, step=step.name
                    )
                )
                await asyncio.sleep(0)

            step_ops = workpiece.layer.get_ops(step, workpiece)
            if not step_ops:
                continue

            # 1. Rotate the ops around its local center.
            wp_angle = workpiece.angle
            if wp_angle != 0:
                wp_w, wp_h = workpiece.size
                cx, cy = wp_w / 2, wp_h / 2
                step_ops.rotate(-wp_angle, cx, cy)

            # 2. Translate to final canonical position on the work area
            step_ops.translate(*workpiece.pos)

            # 3. Convert from canonical (Y-up) to machine-native coords
            if config.machine.y_axis_down:
                step_ops.scale(1, -1)
                step_ops.translate(0, machine_height)

            # 4. Clip to machine boundaries and apply post-transformers
            clipped_ops = step_ops.clip(clip_rect)
            final_ops += clipped_ops * step.passes

        if context:
            context.set_progress(1.0)
            context.flush()
        return final_ops

    def render(
        self, pixels_per_mm_x: int, pixels_per_mm_y: int
    ) -> cairo.ImageSurface:
        """
        Renders the entire document to a new surface.
        """
        surface_width_mm, surface_height_mm = config.machine.dimensions
        width = int(surface_width_mm * pixels_per_mm_x)
        height = int(surface_height_mm * pixels_per_mm_y)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        for workpiece in self.workpieces:
            wp_surface = workpiece.render_for_ops(
                pixels_per_mm_x, pixels_per_mm_y
            )

            if wp_surface:
                pos_x_mm, pos_y_mm = workpiece.pos or (0, 0)
                pos_x = pos_x_mm * pixels_per_mm_x
                pos_y = pos_y_mm * pixels_per_mm_y

                ctx.set_source_surface(wp_surface, pos_x, pos_y)
                ctx.paint()

        return surface

    def save_bitmap(
        self, filename: Path, pixels_per_mm_x: int, pixels_per_mm_y: int
    ):
        surface = self.render(pixels_per_mm_x, pixels_per_mm_y)
        surface.write_to_png(str(filename))
