import math
import logging
from blinker import Signal
from typing import Optional, Tuple, cast, Dict, List
from gi.repository import Graphene, Gdk, Gtk  # type: ignore
from ..core.doc import Doc
from ..core.layer import Layer
from ..core.workpiece import WorkPiece
from ..machine.models.machine import Machine
from ..pipeline.generator import OpsGenerator
from ..undo import (
    SetterCommand,
    ListItemCommand,
    Command,
    ChangePropertyCommand,
)
from .canvas import Canvas, CanvasElement
from .axis import AxisRenderer
from .elements.dot import DotElement
from .elements.step import StepElement
from .elements.workpiece import WorkPieceElement
from .elements.camera_image import CameraImageElement
from .elements.layer import LayerElement
from .aligner import Aligner


logger = logging.getLogger(__name__)


class MoveWorkpiecesLayerCommand(Command):
    """
    An undoable command to move one or more workpieces to a different layer.
    """

    def __init__(
        self,
        canvas: "WorkSurface",
        workpieces: List[WorkPiece],
        new_layer: Layer,
        old_layer: Layer,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.canvas = canvas
        self.workpieces = workpieces
        self.new_layer = new_layer
        self.old_layer = old_layer
        if not name:
            self.name = _("Move to another layer")

    def _move(self, from_layer: Layer, to_layer: Layer):
        """The core logic for moving workpieces, view-first."""
        from_layer_elem = cast(
            Optional[LayerElement], self.canvas.find_by_data(from_layer)
        )
        to_layer_elem = cast(
            Optional[LayerElement], self.canvas.find_by_data(to_layer)
        )

        if not from_layer_elem or not to_layer_elem:
            logger.warning("Could not find layer elements for move operation.")
            return

        # Step 1 & 2: UI-first, flicker-free re-parenting
        elements_to_move = []
        for wp in self.workpieces:
            wp_elem = self.canvas.find_by_data(wp)
            if wp_elem:
                elements_to_move.append(wp_elem)

        for elem in elements_to_move:
            to_layer_elem.add(elem)

        # Enforce correct Z-order immediately after the move.
        from_layer_elem.sort_children_by_z_order()
        to_layer_elem.sort_children_by_z_order()
        self.canvas.queue_draw()

        # Step 3: Update the model
        # The signals fired by these methods will now trigger a non-destructive
        # reconciliation in the LayerElements.
        for wp in self.workpieces:
            from_layer.remove_workpiece(wp)
            to_layer.add_workpiece(wp)

    def execute(self):
        """Executes the command, moving workpieces to the new layer."""
        self._move(self.old_layer, self.new_layer)

    def undo(self):
        """Undoes the command, moving workpieces back to the old layer."""
        self._move(self.new_layer, self.old_layer)


class WorkSurface(Canvas):
    """
    The WorkSurface displays a grid area with WorkPieces and
    WorkPieceOpsElements according to real world dimensions.
    """

    # The minimum allowed zoom level, relative to the "fit-to-view" size
    # (zoom=1.0). 0.1 means you can zoom out until the view is 10% of its
    # "fit" size.
    MIN_ZOOM_FACTOR = 0.1

    # The maximum allowed pixel density when zooming in.
    MAX_PIXELS_PER_MM = 100.0

    def __init__(
        self,
        doc: Doc,
        machine: Optional[Machine],
        cam_visible: bool = False,
        **kwargs,
    ):
        logger.debug("WorkSurface.__init__ called")
        super().__init__(**kwargs)
        self.doc = doc
        self.machine = None  # will be assigned by set_machine() below
        self.ops_generator = OpsGenerator(doc)
        self.zoom_level = 1.0
        self._show_travel_moves = False
        self._workpieces_visible = True
        self.width_mm, self.height_mm = (
            machine.dimensions if machine else (100.0, 100.0)
        )
        self.pixels_per_mm_x = 0.0
        self.pixels_per_mm_y = 0.0
        self._cam_visible = cam_visible
        self._laser_dot_pos_mm = 0.0, 0.0
        self._transform_start_states: Dict[CanvasElement, dict] = {}

        # The root element itself should not clip, allowing its children
        # to draw outside its bounds.
        self.root.clip = False

        y_axis_down = machine.y_axis_down if machine else False
        self._axis_renderer = AxisRenderer(
            width_mm=self.width_mm,
            height_mm=self.height_mm,
            zoom_level=self.zoom_level,
            y_axis_down=y_axis_down,
        )
        self.root.background = 0.8, 0.8, 0.8, 0.1  # light gray background

        # Set theme colors for axis and grid. This will be called on each
        # redraw as well, to handle live theme changes.
        self._update_theme_colors()

        # DotElement size will be set in pixels by WorkSurface
        # Initialize with zero size, size and position will be set in
        # do_size_allocate
        self._laser_dot = DotElement(0, 0, 5.0)
        self.root.add(self._laser_dot)

        # Add scroll event controller for zoom
        self._scroll_controller = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
        )
        self._scroll_controller.connect("scroll", self.on_scroll)
        self.add_controller(self._scroll_controller)

        # Add middle click gesture for panning
        self._pan_gesture = Gtk.GestureDrag.new()
        self._pan_gesture.set_button(Gdk.BUTTON_MIDDLE)
        self._pan_gesture.connect("drag-begin", self.on_pan_begin)
        self._pan_gesture.connect("drag-update", self.on_pan_update)
        self._pan_gesture.connect("drag-end", self.on_pan_end)
        self.add_controller(self._pan_gesture)
        self._pan_start = (0.0, 0.0)

        # This is hacky, but what to do: The EventControllerScroll provides
        # no access to any mouse position, and there is no easy way to
        # get the mouse position in Gtk4. So I have to store it here and
        # track the motion event...
        self._mouse_pos = (0.0, 0.0)

        # Signals for clipboard and duplication operations
        self.cut_requested = Signal()
        self.copy_requested = Signal()
        self.paste_requested = Signal()
        self.duplicate_requested = Signal()
        self.aspect_ratio_changed = Signal()

        # Connect to undo/redo signals from the canvas
        self.move_begin.connect(self._on_any_transform_begin)
        self.move_end.connect(self._on_move_end)
        self.resize_begin.connect(self._on_resize_begin)
        self.resize_end.connect(self._on_resize_end)
        self.rotate_begin.connect(self._on_any_transform_begin)
        self.rotate_end.connect(self._on_rotate_end)
        self.elements_deleted.connect(self._on_elements_deleted)

        # Initialize
        self.set_machine(machine)
        self.aligner = Aligner(self)

        # Connect to the history manager's changed signal to sync the view
        # globally, which is necessary for undo/redo actions triggered
        # outside of this widget.
        self.doc.history_manager.changed.connect(self._on_history_changed)

    def _update_theme_colors(self):
        """
        Reads the current theme colors from the widget's style context
        and applies them to the AxisRenderer.
        """
        style_context = self.get_style_context()

        # Get the foreground color for axes and labels
        found, fg_rgba = style_context.lookup_color("view_fg_color")
        if found:
            self._axis_renderer.set_fg_color(
                (fg_rgba.red, fg_rgba.green, fg_rgba.blue, fg_rgba.alpha)
            )

        # Get the separator color for the grid lines
        found, grid_rgba = style_context.lookup_color("separator_color")
        if found:
            self._axis_renderer.set_grid_color(
                (
                    grid_rgba.red,
                    grid_rgba.green,
                    grid_rgba.blue,
                    grid_rgba.alpha,
                )
            )

    def _on_history_changed(self, sender, **kwargs):
        """
        Called when the undo/redo history changes. This handler acts as a
        synchronizer to fix state timing issues. It re-commits the current
        selection state to ensure all listeners are in sync.
        """
        self._finalize_selection_state()
        self.queue_draw()

    def _on_any_transform_begin(self, sender, elements: List[CanvasElement]):
        """Saves the initial state of elements before any transformation."""
        self._transform_start_states.clear()
        for element in elements:
            if not isinstance(element.data, WorkPiece):
                continue
            workpiece: WorkPiece = element.data
            self._transform_start_states[element] = {
                "pos": workpiece.pos,
                "size": workpiece.size,
                "angle": workpiece.angle,
            }

    def _on_resize_begin(self, sender, elements: List[CanvasElement]):
        """Handles start of a resize, which invalidates Ops."""
        self._on_any_transform_begin(sender, elements)
        self.ops_generator.pause()

    def _on_move_end(self, sender, elements: List[CanvasElement]):
        history = self.doc.history_manager
        with history.transaction(_("Move workpiece(s)")) as t:
            for element in elements:
                if (
                    not isinstance(element.data, WorkPiece)
                    or element not in self._transform_start_states
                ):
                    continue
                workpiece: WorkPiece = element.data
                start_state = self._transform_start_states[element]
                if workpiece.pos and start_state["pos"] != workpiece.pos:
                    t.add(
                        SetterCommand(
                            workpiece,
                            "set_pos",
                            workpiece.pos,
                            start_state["pos"],
                        )
                    )

        self._transform_start_states.clear()

    def _on_rotate_end(self, sender, elements: List[CanvasElement]):
        history = self.doc.history_manager
        with history.transaction(_("Rotate workpiece(s)")) as t:
            for element in elements:
                if (
                    not isinstance(element.data, WorkPiece)
                    or element not in self._transform_start_states
                ):
                    continue
                workpiece: WorkPiece = element.data
                start_state = self._transform_start_states[element]
                if start_state["angle"] != workpiece.angle:
                    t.add(
                        SetterCommand(
                            workpiece,
                            "set_angle",
                            (workpiece.angle,),
                            (start_state["angle"],),
                        )
                    )

        self._transform_start_states.clear()

    def _on_resize_end(self, sender, elements: List[CanvasElement]):
        history = self.doc.history_manager
        with history.transaction(_("Resize workpiece(s)")) as t:
            for element in elements:
                if (
                    not isinstance(element.data, WorkPiece)
                    or element not in self._transform_start_states
                ):
                    continue
                workpiece: WorkPiece = element.data
                start_state = self._transform_start_states[element]
                if workpiece.pos and start_state["pos"] != workpiece.pos:
                    t.add(
                        SetterCommand(
                            workpiece,
                            "set_pos",
                            workpiece.pos,
                            start_state["pos"],
                        )
                    )
                if workpiece.size and start_state["size"] != workpiece.size:
                    t.add(
                        SetterCommand(
                            workpiece,
                            "set_size",
                            workpiece.size,
                            start_state["size"],
                        )
                    )
        self._transform_start_states.clear()
        self.ops_generator.resume()

    def _on_elements_deleted(self, sender, elements: List[CanvasElement]):
        workpieces_to_delete = [
            elem.data for elem in elements if isinstance(elem.data, WorkPiece)
        ]

        if not workpieces_to_delete:
            return

        history = self.doc.history_manager
        with history.transaction(_("Delete workpiece(s)")) as t:
            for wp in workpieces_to_delete:
                cmd = ListItemCommand(
                    owner_obj=self.doc,
                    item=wp,
                    undo_command="add_workpiece",
                    redo_command="remove_workpiece",
                    name=_("Delete workpiece"),
                )
                t.add(cmd)

    def on_button_press(self, gesture, n_press: int, x: float, y: float):
        # Let the parent Canvas handle the event first. It will update the
        # selection, set self._active_elem, and determine if a resize is
        # starting.
        super().on_button_press(gesture, n_press, x, y)

        # After the click, check if a new workpiece is active.
        active_wp = self.get_active_workpiece()
        if active_wp and active_wp.layer:
            # If the workpiece's layer is not the document's active layer,
            # create an undoable command to change it.
            if active_wp.layer != self.doc.active_layer:
                cmd = ChangePropertyCommand(
                    target=self.doc,
                    property_name="active_layer",
                    new_value=active_wp.layer,
                    name=_("Select Layer"),
                )
                # Using execute() adds it to the undo stack.
                self.doc.history_manager.execute(cmd)

        # If a resize operation has started on a WorkPieceElement, hide the
        # corresponding ops elements to improve performance.
        if self._resizing and (self._active_elem or self._selection_group):
            elements_in_transform = self.get_selected_elements()
            for element in elements_in_transform:
                if not isinstance(element.data, WorkPiece):
                    continue
                workpiece_data = element.data
                for step_elem in self.find_by_type(StepElement):
                    ops_elem = step_elem.find_by_data(workpiece_data)
                    if ops_elem:
                        ops_elem.set_visible(False)

    def on_button_release(self, gesture, x: float, y: float):
        # Before the parent class resets the resizing state, check if a resize
        # was in progress on a WorkPieceElement.
        workpieces_to_update = []
        if self._resizing and (self._active_elem or self._selection_group):
            elements_in_transform = self.get_selected_elements()
            for element in elements_in_transform:
                if isinstance(element.data, WorkPiece):
                    workpieces_to_update.append(element.data)

        # Let the parent class finish the drag/resize operation.
        super().on_button_release(gesture, x, y)

        # If a resize has just finished, make the ops visible again and
        # trigger a re-allocation and re-render to reflect the new size.
        if workpieces_to_update:
            for workpiece in workpieces_to_update:
                for step_elem in self.find_by_type(StepElement):
                    ops_elem = step_elem.find_by_data(workpiece)
                    if ops_elem:
                        ops_elem.set_visible(True)

    def set_machine(self, machine: Optional[Machine]):
        """
        Updates the WorkSurface to use a new machine instance. This handles
        disconnecting from the old machine's signals, connecting to the new
        one's, and performing a full reset of the view.
        """
        if self.machine is machine:
            return

        # Disconnect from the old machine's signals
        if self.machine:
            self.machine.changed.disconnect(self._on_machine_changed)

        # Update the machine reference
        self.machine = machine

        # Connect to the new machine's signals
        if self.machine:
            self.machine.changed.connect(self._on_machine_changed)
            self.reset_view()

    def set_pan(self, pan_x_mm: float, pan_y_mm: float):
        """Sets the pan position in mm and updates the axis renderer."""
        self._axis_renderer.set_pan_x_mm(pan_x_mm)
        self._axis_renderer.set_pan_y_mm(pan_y_mm)
        self._recalculate_sizes()
        self.queue_draw()

    def _get_base_pixels_per_mm(self) -> Tuple[float, float]:
        """Calculates the pixels/mm for a zoom level of 1.0 (fit-to-view)."""
        width, height_pixels = self.get_width(), self.get_height()
        if not all([width, height_pixels, self.width_mm, self.height_mm]):
            return 1.0, 1.0  # Avoid division by zero at startup

        y_axis_pixels = self._axis_renderer.get_y_axis_width()
        x_axis_height = self._axis_renderer.get_x_axis_height()
        right_margin = math.ceil(y_axis_pixels / 2)
        top_margin = math.ceil(x_axis_height / 2)
        content_width_px = float(width - y_axis_pixels - right_margin)
        content_height_px = float(height_pixels - x_axis_height - top_margin)

        base_ppm_x = (
            content_width_px / self.width_mm if self.width_mm > 0 else 0
        )
        base_ppm_y = (
            content_height_px / self.height_mm if self.height_mm > 0 else 0
        )
        return base_ppm_x, base_ppm_y

    def set_zoom(self, zoom_level: float):
        """
        Sets the zoom level and updates the axis renderer.
        The caller is responsible for ensuring the zoom_level is clamped.
        """
        self.zoom_level = zoom_level
        self._axis_renderer.set_zoom(self.zoom_level)
        self.root.mark_dirty(recursive=True)
        self.do_size_allocate(self.get_width(), self.get_height(), 0)
        self.queue_draw()

    def set_size(self, width_mm: float, height_mm: float):
        """
        Sets the real-world size of the work surface in mm
        and updates related properties.
        """
        self.width_mm = width_mm
        self.height_mm = height_mm
        self._axis_renderer.set_width_mm(self.width_mm)
        self._axis_renderer.set_height_mm(self.height_mm)
        self.queue_draw()

    def get_size(self) -> Tuple[float, float]:
        """Returns the size of the work surface in mm."""
        return self.width_mm, self.height_mm

    def on_motion(self, gesture, x: float, y: float):
        self._mouse_pos = x, y
        return super().on_motion(gesture, x, y)

    def _widget_px_to_view_mm(
        self, x_px: float, y_px: float
    ) -> Tuple[float, float]:
        """
        Centralized conversion from absolute widget pixels to panned/zoomed
        mm (view space). This is the only place that should handle the
        y_axis_down flip.
        """
        y_axis_width = self._axis_renderer.get_y_axis_width()
        x_axis_height = self._axis_renderer.get_x_axis_height()
        height = self.get_height()
        ppm_x = self.pixels_per_mm_x or 1
        ppm_y = self.pixels_per_mm_y or 1
        top_margin = math.ceil(x_axis_height / 2)

        x_mm = (x_px - y_axis_width) / ppm_x
        if self._axis_renderer.y_axis_down:
            y_mm = (y_px - top_margin) / ppm_y
        else:
            y_mm = (height - x_axis_height - y_px) / ppm_y
        return x_mm, y_mm

    def pixel_to_mm(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """
        Converts absolute widget pixel coordinates to absolute machine mm.
        """
        relative_x_mm, relative_y_mm = self._widget_px_to_view_mm(x_px, y_px)
        return (
            relative_x_mm + self._axis_renderer.pan_x_mm,
            relative_y_mm + self._axis_renderer.pan_y_mm,
        )

    def workpiece_coords_to_element_coords(
        self, workpiece: WorkPiece
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Converts workpiece model data (Y-up) to element coords
        (top-left, px) relative to the content area.
        """
        pos_mm = workpiece.pos or (0, 0)
        size_mm = workpiece.size or (0, 0)
        ppm_x = self.pixels_per_mm_x or 1
        ppm_y = self.pixels_per_mm_y or 1

        # Convert size
        width_px = size_mm[0] * ppm_x
        height_px = size_mm[1] * ppm_y

        # Get workpiece top-left corner in absolute canonical (Y-up) mm
        top_left_x_mm = pos_mm[0]
        top_left_y_mm = pos_mm[1] + size_mm[1]

        # Convert absolute canonical mm to content-relative pixels
        x_px = top_left_x_mm * ppm_x
        y_px = self.root.height - (top_left_y_mm * ppm_y)

        return (x_px, y_px), (width_px, height_px)

    def element_coords_to_workpiece_coords(
        self, pos_px: Tuple[float, float], size_px: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Converts element coords (top-left, content-relative px) to workpiece
        model coords (bottom-left, canonical Y-up mm).
        """
        ppm_x = self.pixels_per_mm_x or 1
        ppm_y = self.pixels_per_mm_y or 1

        # Convert size
        width_mm = size_px[0] / ppm_x
        height_mm = size_px[1] / ppm_y

        # Convert content-relative top-left pixel to absolute canonical mm
        abs_tl_x_mm = pos_px[0] / ppm_x
        abs_tl_y_mm = (self.root.height - pos_px[1]) / ppm_y

        # Convert absolute top-left mm to absolute bottom-left mm
        x_mm = abs_tl_x_mm
        y_mm = abs_tl_y_mm - height_mm

        return (x_mm, y_mm), (width_mm, height_mm)

    def on_scroll(self, controller, dx: float, dy: float):
        """Handles the scroll event for zoom."""
        zoom_speed = 0.1

        # 1. Calculate a desired new zoom level based on scroll direction
        if dy > 0:  # Scroll down - zoom out
            desired_zoom = self.zoom_level * (1 - zoom_speed)
        else:  # Scroll up - zoom in
            desired_zoom = self.zoom_level * (1 + zoom_speed)

        # 2. Get the base "fit-to-view" pixel density (for zoom = 1.0)
        base_ppm_x, base_ppm_y = self._get_base_pixels_per_mm()
        if base_ppm_x <= 0 or base_ppm_y <= 0:
            return  # Cannot calculate zoom limits yet (e.g., at startup)

        # Use the smaller base density for consistent limit calculations
        base_ppm = min(base_ppm_x, base_ppm_y)

        # 3. Calculate the pixel density limits
        min_ppm = base_ppm * self.MIN_ZOOM_FACTOR
        max_ppm = self.MAX_PIXELS_PER_MM

        # 4. Calculate the target density and clamp it within our limits
        target_ppm = base_ppm * desired_zoom
        clamped_ppm = max(min_ppm, min(target_ppm, max_ppm))

        # 5. Convert the valid, clamped density back into a final zoom level
        final_zoom = clamped_ppm / base_ppm
        if abs(final_zoom - self.zoom_level) < 1e-9:
            return

        # 6. Calculate pan adjustment to zoom around the mouse cursor
        mouse_x_px, mouse_y_px = self._mouse_pos
        # The real-world point under the cursor before the zoom
        focus_x_mm, focus_y_mm = self.pixel_to_mm(mouse_x_px, mouse_y_px)

        # Temporarily update ppm to calculate the new view coordinates
        new_ppm_x = base_ppm_x * final_zoom
        new_ppm_y = base_ppm_y * final_zoom

        if new_ppm_x > 0 and new_ppm_y > 0:
            # Re-calculate what the view coordinates would be with the new zoom
            y_axis_width = self._axis_renderer.get_y_axis_width()
            x_axis_height = self._axis_renderer.get_x_axis_height()
            height = self.get_height()
            top_margin = math.ceil(x_axis_height / 2)

            new_view_x_mm = (mouse_x_px - y_axis_width) / new_ppm_x
            if self._axis_renderer.y_axis_down:
                new_view_y_mm = (mouse_y_px - top_margin) / new_ppm_y
            else:
                new_view_y_mm = (
                    height - x_axis_height - mouse_y_px
                ) / new_ppm_y

            # The new pan is the difference between the fixed world point
            # and its new panned position
            new_pan_x_mm = focus_x_mm - new_view_x_mm
            new_pan_y_mm = focus_y_mm - new_view_y_mm
            self.set_pan(new_pan_x_mm, new_pan_y_mm)

        # 7. Apply the final, clamped zoom level.
        self.set_zoom(final_zoom)

    def _recalculate_sizes(self):
        origin_x, origin_y = self._axis_renderer.get_origin()
        content_width, content_height = self._axis_renderer.get_content_size()

        # Set the root element's size directly in pixels
        # The root element's origin is always its top-left corner
        if self._axis_renderer.y_axis_down:
            self.root.set_pos(origin_x, origin_y)
        else:
            self.root.set_pos(origin_x, origin_y - content_height)
        self.root.set_size(content_width, content_height)

        # Update WorkSurface's internal pixel dimensions based on content area
        self.pixels_per_mm_x, self.pixels_per_mm_y = (
            self._axis_renderer.get_pixels_per_mm()
        )

        # Update children to match the new content area size
        for elem in self.find_by_type(LayerElement):
            elem.set_size(content_width, content_height)
        for elem in self.find_by_type(CameraImageElement):
            elem.set_size(content_width, content_height)
        # Re-position laser dot based on new pixel dimensions
        # By storing the dot's mm position, we avoid a fragile pixel->mm->pixel
        # conversion during zoom/pan, which could cause drift. We simply
        # re-apply the true mm position to the newly sized/panned view.
        self.set_laser_dot_position(
            self._laser_dot_pos_mm[0], self._laser_dot_pos_mm[1]
        )

    def do_size_allocate(self, width: int, height: int, baseline: int):
        """Handles canvas size allocation in pixels."""
        # Calculate grid bounds using AxisRenderer
        self._axis_renderer.set_width_px(width)
        self._axis_renderer.set_height_px(height)
        self._recalculate_sizes()
        self.root.allocate()

    def set_show_travel_moves(self, show: bool):
        """Sets whether to display travel moves and triggers re-rendering."""
        if self._show_travel_moves != show:
            self._show_travel_moves = show
            # Propagate the change to all existing StepElements
            for elem in self.find_by_type(StepElement):
                elem = cast(StepElement, elem)
                elem.set_show_travel_moves(show)

    def _create_and_add_layer_element(self, layer: "Layer"):
        """Creates a new LayerElement and adds it to the canvas root."""
        logger.debug(f"Adding new LayerElement for '{layer.name}'")
        layer_elem = LayerElement(layer=layer, canvas=self)

        # A LayerElement is a container that spans the entire content area
        content_width, content_height = self._axis_renderer.get_content_size()
        layer_elem.set_size(content_width, content_height)

        self.root.add(layer_elem)

        # Now populate the new layer element with its children
        layer_elem.sync_with_model(layer)

    def update_from_doc(self, doc: Doc):
        """
        Synchronizes the canvas elements with the document model.

        This method ensures that the layers and their contents (workpieces,
        steps) displayed on the canvas perfectly match the state of the
        document's data model. It also reorders the LayerElements to match
        the Z-order of the layers in the document.
        """
        self.doc = doc
        self.ops_generator.doc = doc

        # --- Step 1: Add and Remove LayerElements ---
        doc_layers_set = set(doc.layers)
        current_elements_on_canvas = {
            elem.data: elem for elem in self.find_by_type(LayerElement)
        }

        # Remove elements for layers that are no longer in the doc
        for layer, elem in current_elements_on_canvas.items():
            if layer not in doc_layers_set:
                elem.remove()

        # Add elements for new layers that are not yet on the canvas
        for layer in doc.layers:
            if layer not in current_elements_on_canvas:
                self._create_and_add_layer_element(layer)

        # --- Step 2: Reorder LayerElements for Z-stacking ---
        # The first layer in the list is at the bottom (drawn first).
        # The last layer is at the top (drawn last).
        layer_order_map = {layer: i for i, layer in enumerate(doc.layers)}

        def sort_key(element: CanvasElement):
            """
            Sort key for root's children. Camera at bottom, then dot,
            then layers.
            """
            if isinstance(element, LayerElement):
                # LayerElements are ordered according to the doc.layers list.
                return layer_order_map.get(element.data, len(layer_order_map))
            if isinstance(element, CameraImageElement):
                # Camera images are at the very bottom.
                return -2
            # Other elements (like the laser dot) are above the camera but
            # below layers.
            return -1

        self.root.children.sort(key=sort_key)

        self.queue_draw()

    def set_laser_dot_visible(self, visible=True):
        self._laser_dot.set_visible(visible)
        self.queue_draw()

    def mm_to_element_coords(
        self, x_mm: float, y_mm: float, element: CanvasElement
    ) -> Tuple[float, float]:
        """
        Converts absolute machine mm coordinates to pixel coordinates relative
        to the given element's top-left corner.
        """
        # Part 1: Calculate the point's coordinates relative to the root
        # element.
        ppm_x = self.pixels_per_mm_x or 1
        ppm_y = self.pixels_per_mm_y or 1

        # Step 1: Convert the incoming machine coordinates into the `root`
        # element's canonical Y-up coordinate space.
        x_mm_canonical: float
        y_mm_canonical: float
        if self._axis_renderer.y_axis_down:
            y_mm_canonical = self.height_mm - y_mm
        else:
            y_mm_canonical = y_mm
        x_mm_canonical = x_mm

        # Step 2: Scale the canonical mm-coordinates to get pixel coordinates
        # within the root's virtual space. Pan is handled by moving the root
        # element, so no pan subtraction is needed here.
        center_x_px = x_mm_canonical * ppm_x
        center_y_px = y_mm_canonical * ppm_y

        # Step 3: Convert the canonical Y-up pixel coordinates to the root's
        # internal drawing coordinates, which have their origin at the
        # top-left.
        point_x_in_root = center_x_px
        point_y_in_root = self.root.height - center_y_px

        # Part 2: Calculate the target element's coordinates relative to the
        # root by walking up the element tree.
        elem_x_in_root = 0.0
        elem_y_in_root = 0.0
        current = element
        while current and current is not self.root:
            elem_x_in_root += current.x
            elem_y_in_root += current.y
            current = current.parent

        # Part 3: The final coordinates are the point's position in the root
        # minus the element's position in the root.
        return (
            point_x_in_root - elem_x_in_root,
            point_y_in_root - elem_y_in_root,
        )

    def set_laser_dot_position(self, x_mm: float, y_mm: float):
        """Sets the laser dot position in real-world mm."""
        self._laser_dot_pos_mm = x_mm, y_mm

        dot_parent = self._laser_dot.parent
        if not dot_parent:
            return  # Cannot position if not in the tree

        # Convert machine mm to pixel coordinates relative to the dot's parent.
        center_x_rel, center_y_rel = self.mm_to_element_coords(
            x_mm, y_mm, dot_parent
        )

        # Calculate the top-left corner of the dot's bounding box
        # from its center point.
        dot_width_px = self._laser_dot.width
        dot_height_px = self._laser_dot.height

        self._laser_dot.set_pos(
            round(center_x_rel - dot_width_px / 2),
            round(center_y_rel - dot_height_px / 2),
        )
        self.queue_draw()

    def remove_all(self):
        # Clear all children except the fixed ones
        children_to_remove = [
            c
            for c in self.root.children
            if not isinstance(c, (CameraImageElement, DotElement))
        ]
        for child in children_to_remove:
            child.remove()
        self.queue_draw()

    def find_by_type(self, thetype):
        """
        Search recursively through the root's children
        """
        return self.root.find_by_type(thetype)

    def set_workpieces_visible(self, visible=True):
        self._workpieces_visible = visible
        for wp_elem in self.find_by_type(WorkPieceElement):
            wp_elem.set_visible(visible)
        self.queue_draw()

    def set_camera_image_visibility(self, visible: bool):
        self._cam_visible = visible
        for elem in self.find_by_type(CameraImageElement):
            camera_elem = cast(CameraImageElement, elem)
            camera_elem.set_visible(visible and camera_elem.camera.enabled)
        self.queue_draw()

    def _on_machine_changed(self, machine: Optional[Machine]):
        """
        Handles incremental updates from the currently-assigned machine model.
        If core properties like dimensions or axis direction change, it
        performs a full view reset. Otherwise, it syncs other properties like
        cameras.
        """
        if not machine:
            return

        # Check for changes that require a full view reset. A change to either
        # dimensions or y-axis orientation invalidates the current pan, zoom,
        # and all calculated coordinates.
        size_changed = machine.dimensions != (self.width_mm, self.height_mm)
        y_axis_changed = machine.y_axis_down != self._axis_renderer.y_axis_down

        if size_changed or y_axis_changed:
            self.reset_view()
        else:
            # No major reset needed, but other properties like the list of
            # cameras might have changed.
            self._sync_camera_elements(machine)

    def reset_view(self):
        """
        Resets the view to fit the given machine's properties, including a
        full reset of pan, zoom, and size. Also syncs camera elements.
        """
        if not self.machine:
            return
        logger.debug(
            f"Resetting view for machine '{self.machine.name}' "
            f"with dims={self.machine.dimensions} and "
            f"y_down={self.machine.y_axis_down}"
        )
        new_dimensions = self.machine.dimensions
        self.set_size(new_dimensions[0], new_dimensions[1])
        self.set_pan(0.0, 0.0)
        self.set_zoom(1.0)
        self._axis_renderer.set_y_axis_down(self.machine.y_axis_down)
        # _recalculate_sizes must be called after other properties are set,
        # especially after y_axis_down is changed, as it affects all
        # coordinate calculations.
        self._recalculate_sizes()

        new_ratio = (
            new_dimensions[0] / new_dimensions[1]
            if new_dimensions[1] > 0
            else 1.0
        )
        self.aspect_ratio_changed.send(self, ratio=new_ratio)
        self._sync_camera_elements(self.machine)
        self.queue_draw()

    def _sync_camera_elements(self, machine: Machine):
        """Adds, removes, and updates camera elements on the canvas."""
        # Get current camera elements on the canvas
        current_camera_elements = {
            cast(CameraImageElement, elem).camera: elem
            for elem in self.find_by_type(CameraImageElement)
        }
        new_cameras = machine.cameras

        cameras_on_canvas = set(current_camera_elements.keys())
        cameras_in_model = set(new_cameras)

        # If there are no changes, do nothing.
        if cameras_on_canvas == cameras_in_model:
            return

        logger.debug("Syncing camera elements.")

        # Add new camera elements
        for camera in cameras_in_model - cameras_on_canvas:
            camera_image_elem = CameraImageElement(camera)
            camera_image_elem.set_visible(self._cam_visible and camera.enabled)
            self.root.insert(0, camera_image_elem)
            logger.debug(f"Added CameraImageElement for camera {camera.name}")

        # Remove camera elements that no longer exist in the machine
        for camera_instance in cameras_on_canvas - cameras_in_model:
            elem = current_camera_elements[camera_instance]
            elem.remove()
            logger.debug(
                f"Removed CameraImageElement for camera {camera_instance.name}"
            )

    def do_snapshot(self, snapshot):
        # Update theme colors right before drawing to catch any live changes.
        self._update_theme_colors()

        # Create a Cairo context for the snapshot
        width, height = self.get_width(), self.get_height()
        bounds = Graphene.Rect().init(0, 0, width, height)
        ctx = snapshot.append_cairo(bounds)

        # Draw grid, axis, and labels first, so they are in the background.
        self._axis_renderer.draw_grid(ctx)
        self._axis_renderer.draw_axes_and_labels(ctx)

        # Use the parent Canvas's recursive rendering.
        super().do_snapshot(snapshot)

    def on_key_pressed(
        self, controller, keyval: int, keycode: int, state: Gdk.ModifierType
    ) -> bool:
        """Handles key press events for the work surface."""
        # Reset pan and zoom with '1'
        if keyval == Gdk.KEY_1:
            self.reset_view()
            return True  # Event handled

        is_ctrl = bool(state & Gdk.ModifierType.CONTROL_MASK)

        # Handle moving workpiece to another layer
        if is_ctrl and (
            keyval == Gdk.KEY_Page_Up or keyval == Gdk.KEY_Page_Down
        ):
            selected_wps = self.get_selected_workpieces()
            if not selected_wps:
                return True

            layers = self.doc.layers
            if len(layers) <= 1:
                return True

            current_layer = selected_wps[0].layer
            if not current_layer:
                return True

            current_index = layers.index(current_layer)
            direction = -1 if keyval == Gdk.KEY_Page_Up else 1
            new_index = (current_index + direction + len(layers)) % len(layers)
            new_layer = layers[new_index]

            cmd = MoveWorkpiecesLayerCommand(
                self, selected_wps, new_layer, current_layer
            )
            self.doc.history_manager.execute(cmd)

            return True

        # Handle clipboard and duplication
        if is_ctrl:
            selected_wps = self.get_selected_workpieces()
            if keyval == Gdk.KEY_x:
                if selected_wps:
                    self.cut_requested.send(self, workpieces=selected_wps)
                    return True
            elif keyval == Gdk.KEY_c:
                if selected_wps:
                    self.copy_requested.send(self, workpieces=selected_wps)
                    return True
            elif keyval == Gdk.KEY_v:
                self.paste_requested.send(self)
                return True
            elif keyval == Gdk.KEY_d:
                if selected_wps:
                    self.duplicate_requested.send(
                        self, workpieces=selected_wps
                    )
                    return True
            elif keyval == Gdk.KEY_a:
                # Select all workpieces
                all_workpieces = self.doc.workpieces
                if all_workpieces:
                    self.select_workpieces(all_workpieces)

        # Handle arrow key movement for selected workpieces
        is_shift = bool(state & Gdk.ModifierType.SHIFT_MASK)

        move_amount_mm = 1.0
        if is_shift:
            move_amount_mm *= 10
        elif is_ctrl:
            move_amount_mm *= 0.1

        move_amount_x_mm = 0.0
        move_amount_y_mm = 0.0

        if keyval == Gdk.KEY_Up:
            move_amount_y_mm = move_amount_mm
        elif keyval == Gdk.KEY_Down:
            move_amount_y_mm = -move_amount_mm
        elif keyval == Gdk.KEY_Left:
            move_amount_x_mm = -move_amount_mm
        elif keyval == Gdk.KEY_Right:
            move_amount_x_mm = move_amount_mm

        if move_amount_x_mm != 0 or move_amount_y_mm != 0:
            selected_wps = self.get_selected_workpieces()
            if not selected_wps:
                return True  # Consume event but do nothing

            history = self.doc.history_manager
            with history.transaction(_("Move the workpiece")) as t:
                for wp in selected_wps:
                    old_pos = wp.pos
                    if old_pos:
                        # Arrow keys always manipulate the canonical model
                        # coordinates. "Up" arrow always increases the Y value.
                        new_pos = (
                            old_pos[0] + move_amount_x_mm,
                            old_pos[1] + move_amount_y_mm,
                        )
                        cmd = SetterCommand(
                            wp, "set_pos", new_args=new_pos, old_args=old_pos
                        )
                        t.execute(cmd)
            return True

        # Propagate to parent Canvas for its default behavior (e.g., Shift/
        # Ctrl)
        return super().on_key_pressed(controller, keyval, keycode, state)

    def on_pan_begin(self, gesture, x: float, y: float):
        self._pan_start = (
            self._axis_renderer.pan_x_mm,
            self._axis_renderer.pan_y_mm,
        )

    def on_pan_update(self, gesture, x: float, y: float):
        # Calculate pan offset based on drag delta
        offset = gesture.get_offset()
        new_pan_x_mm = self._pan_start[0] - offset.x / self.pixels_per_mm_x

        # For Y-panning, dragging down (positive offset.y) should always
        # move the content "up" on screen.
        if self._axis_renderer.y_axis_down:
            # In a Y-down view, moving content "up" means panning to lower
            # Y values.
            new_pan_y_mm = self._pan_start[1] - offset.y / self.pixels_per_mm_y
        else:
            # In a Y-up view, moving content "up" means panning to higher
            # Y values.
            new_pan_y_mm = self._pan_start[1] + offset.y / self.pixels_per_mm_y
        self.set_pan(new_pan_x_mm, new_pan_y_mm)

    def on_pan_end(self, gesture, x: float, y: float):
        pass

    def get_active_workpiece(self) -> Optional[WorkPiece]:
        active_elem = self.get_active_element()
        if active_elem and isinstance(active_elem.data, WorkPiece):
            return active_elem.data
        return None

    def get_selected_workpieces(self) -> List[WorkPiece]:
        selected_workpieces = []
        for elem in self.get_selected_elements():
            if isinstance(elem.data, WorkPiece):
                selected_workpieces.append(elem.data)
        return selected_workpieces

    def select_workpieces(self, workpieces_to_select: List[WorkPiece]):
        """
        Clears the current selection and selects the canvas elements
        corresponding to the given list of WorkPiece objects.
        """
        self.root.unselect_all()
        uids_to_select = {wp.uid for wp in workpieces_to_select}
        self._active_elem = None

        # Iterate through the canvas elements to find the ones to select
        for elem in self.find_by_type(WorkPieceElement):
            if (
                elem.data
                and hasattr(elem.data, "uid")
                and elem.data.uid in uids_to_select
            ):
                elem.selected = True
                # The last one in the list becomes the active one
                self._active_elem = elem

        self._finalize_selection_state()
        self.queue_draw()

    def center_horizontally(self):
        """
        Horizontally aligns selected workpieces.
        - One item: centered on the canvas.
        - Multiple items: centered on their collective bounding box center.
        """
        self.aligner.center_horizontally()

    def center_vertically(self):
        """
        Vertically aligns selected workpieces.
        - One item: centered on the canvas.
        - Multiple items: centered on their collective bounding box center.
        """
        self.aligner.center_vertically()

    def align_left(self):
        """
        Aligns the left edge of selected items.
        - One item: aligned to the left edge of the canvas.
        - Multiple items: aligned to the left edge of their collective bbox.
        """
        self.aligner.align_left()

    def align_right(self):
        """
        Aligns the right edge of selected items.
        - One item: aligned to the right edge of the canvas.
        - Multiple items: aligned to the right edge of their collective bbox.
        """
        self.aligner.align_right()

    def align_top(self):
        """
        Aligns the top edge of selected items.
        - One item: aligned to the top edge of the canvas.
        - Multiple items: aligned to the top edge of their collective bbox.
        """
        self.aligner.align_top()

    def align_bottom(self):
        """
        Aligns the bottom edge of selected items.
        - One item: aligned to the bottom edge of the canvas.
        - Multiple items: aligned to the bottom edge of their collective bbox.
        """
        self.aligner.align_bottom()

    def spread_horizontally(self):
        """
        Horizontally distributes selected items evenly over the work surface.
        """
        self.aligner.spread_horizontally()

    def spread_vertically(self):
        """
        Vertically distributes selected items evenly over the work surface.
        """
        self.aligner.spread_vertically()
