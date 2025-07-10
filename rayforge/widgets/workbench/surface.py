import math
import logging
from typing import Optional, Tuple, cast
from gi.repository import Graphene, Gdk, Gtk  # type: ignore
from ...models.doc import Doc
from ...models.workpiece import WorkPiece
from ..canvas import Canvas, CanvasElement
from .axis import AxisRenderer
from .dotelem import DotElement
from .workstepelem import WorkStepElement
from .workpieceelem import WorkPieceElement
from typing import List


logger = logging.getLogger(__name__)


class WorkSurface(Canvas):
    """
    The WorkSurface displays a grid area with WorkPieces and
    WorkPieceOpsElements according to real world dimensions.
    """

    def __init__(self, **kwargs):
        logger.debug("WorkSurface.__init__ called")
        super().__init__(**kwargs)
        self.zoom_level = 1.0
        self.show_travel_moves = False
        self.width_mm = 100.0
        self.height_mm = 100.0
        self.pixels_per_mm_x = 0.0
        self.pixels_per_mm_y = 0.0

        self.axis_renderer = AxisRenderer(
            width_mm=self.width_mm,
            height_mm=self.height_mm,
            zoom_level=self.zoom_level,
        )
        self.root.background = 0.8, 0.8, 0.8, 0.1  # light gray background

        # These elements will be sized and positioned in pixels by WorkSurface
        self.workpiece_elements = CanvasElement(0, 0, 0, 0, selectable=False)
        self.root.add(self.workpiece_elements)

        # DotElement size will be set in pixels by WorkSurface
        # Initialize with zero size, size and position will be set in
        # do_size_allocate
        self.laser_dot = DotElement(0, 0, 0, 0)
        self.root.add(self.laser_dot)

        # Add scroll event controller for zoom
        self.scroll_controller = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
        )
        self.scroll_controller.connect("scroll", self.on_scroll)
        self.add_controller(self.scroll_controller)

        # Add middle click gesture for panning
        self.pan_gesture = Gtk.GestureDrag.new()
        self.pan_gesture.set_button(Gdk.BUTTON_MIDDLE)
        self.pan_gesture.connect("drag-begin", self.on_pan_begin)
        self.pan_gesture.connect("drag-update", self.on_pan_update)
        self.pan_gesture.connect("drag-end", self.on_pan_end)
        self.add_controller(self.pan_gesture)
        self.pan_start = 0, 0

        # This is hacky, but what to do: The EventControllerScroll provides
        # no access to any mouse position, and there is no easy way to
        # get the mouse position in Gtk4. So I have to store it here and
        # track the motion event...
        self.mouse_pos = 0, 0
        self.doc: Optional[Doc] = None
        self.elem_removed.connect(self._on_elem_removed)

    def set_pan(self, pan_x_mm: float, pan_y_mm: float):
        """Sets the pan position in mm and updates the axis renderer."""
        self.axis_renderer.set_pan_x_mm(pan_x_mm)
        self.axis_renderer.set_pan_y_mm(pan_y_mm)
        self._recalculate_sizes()
        self.queue_draw()

    def set_zoom(self, zoom_level: float):
        """Sets the zoom level and updates the axis renderer."""
        self.zoom_level = max(0.5, min(zoom_level, 1.5))
        self.axis_renderer.set_zoom(self.zoom_level)
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
        self.axis_renderer.set_width_mm(self.width_mm)
        self.axis_renderer.set_height_mm(self.height_mm)
        self.queue_draw()

    def on_motion(self, gesture, x: int, y: int):
        self.mouse_pos = x, y
        return super().on_motion(gesture, x, y)

    def pixel_to_mm(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Converts pixel coordinates to real-world mm."""
        height_pixels = self.get_height()
        y_axis_pixels = self.axis_renderer.get_y_axis_width()
        x_axis_height = self.axis_renderer.get_x_axis_height()
        top_margin = math.ceil(x_axis_height / 2)
        x_mm = self.axis_renderer.pan_x_mm + (
            (x_px - y_axis_pixels) / self.pixels_per_mm_x
        )
        y_mm = (
            self.axis_renderer.pan_y_mm
            + (height_pixels - y_px - top_margin)
            / self.pixels_per_mm_y
        )
        return x_mm, y_mm

    def on_scroll(self, controller, dx: float, dy: float):
        """Handles the scroll event for zoom."""
        zoom_speed = 0.1

        # Adjust zoom level
        if dy > 0:  # Scroll down - zoom out
            new_zoom = self.zoom_level * (1 - zoom_speed)
        else:  # Scroll up - zoom in
            new_zoom = self.zoom_level * (1 + zoom_speed)

        # Get current mouse position in mm
        mouse_x_px, mouse_y_px = self.mouse_pos
        focus_x_mm, focus_y_mm = self.pixel_to_mm(mouse_x_px, mouse_y_px)

        # Calculate new content size and scaling
        width, height_pixels = self.get_width(), self.get_height()
        y_axis_pixels = self.axis_renderer.get_y_axis_width()
        x_axis_height = self.axis_renderer.get_x_axis_height()
        right_margin = math.ceil(y_axis_pixels / 2)
        top_margin = math.ceil(x_axis_height / 2)
        content_width_px = width - y_axis_pixels - right_margin
        content_height_px = height_pixels - x_axis_height - top_margin
        new_pixels_per_mm_x = (
            content_width_px / self.width_mm * new_zoom
            if self.width_mm > 0
            else 0
        )
        new_pixels_per_mm_y = (
            content_height_px / self.height_mm * new_zoom
            if self.height_mm > 0
            else 0
        )

        # Adjust pan to keep focus point under cursor
        new_pan_x_mm = (
            focus_x_mm - (mouse_x_px - y_axis_pixels) / new_pixels_per_mm_x
        )
        new_pan_y_mm = (
            focus_y_mm
            - (height_pixels - mouse_y_px - top_margin) / new_pixels_per_mm_y
        )

        # Update rendering
        self.set_zoom(new_zoom)
        self.set_pan(new_pan_x_mm, new_pan_y_mm)

    def _recalculate_sizes(self):
        origin_x, origin_y = self.axis_renderer.get_origin()
        content_width, content_height = self.axis_renderer.get_content_size()

        # Set the root element's size directly in pixels
        self.root.set_pos(origin_x, origin_y - content_height)
        self.root.set_size(content_width, content_height)

        # Update WorkSurface's internal pixel dimensions based on content area
        self.pixels_per_mm_x, self.pixels_per_mm_y = (
            self.axis_renderer.get_pixels_per_mm()
        )

        # Update the workpiece element group and WorkStepElement group sizes:
        # they should always match root group size
        content_width, content_height = self.axis_renderer.get_content_size()
        self.workpiece_elements.set_size(content_width, content_height)
        for elem in self.find_by_type(WorkStepElement):
            elem.set_size(content_width, content_height)

        # Update laser dot size based on new pixel dimensions and its mm radius
        dot_radius_mm = self.laser_dot.radius_mm
        dot_diameter_px = round(2 * dot_radius_mm * self.pixels_per_mm_x)
        self.laser_dot.set_size(dot_diameter_px, dot_diameter_px)

        # Re-position laser dot based on new pixel dimensions
        current_dot_pos_px = self.laser_dot.pos_abs()
        current_dot_pos_mm = self.laser_dot.pixel_to_mm(*current_dot_pos_px)
        self.set_laser_dot_position(*current_dot_pos_mm)

    def do_size_allocate(self, width: int, height: int, baseline: int):
        """Handles canvas size allocation in pixels."""
        # Calculate grid bounds using AxisRenderer
        self.axis_renderer.set_width_px(width)
        self.axis_renderer.set_height_px(height)
        self._recalculate_sizes()
        self.root.allocate()

    def set_show_travel_moves(self, show: bool):
        """Sets whether to display travel moves and triggers a redraw."""
        if self.show_travel_moves != show:
            self.show_travel_moves = show
            # Mark elements dirty that depend on this setting
            for elem in self.find_by_type(WorkStepElement):
                elem.mark_dirty()
            self.queue_draw()

    def _on_elem_removed(self, sender, child):
        if not self.doc or not isinstance(child.data, WorkPiece):
            return
        self.doc.remove_workpiece(child.data)
        self.update_from_doc(self.doc)

    def update_from_doc(self, doc: Doc):
        self.doc = doc

        # Remove anything from the canvas that no longer exists.
        for elem in self.find_by_type(WorkStepElement):
            if elem.data not in doc.workplan:
                elem.remove()
        for elem in self.find_by_type(WorkPieceElement):
            if elem.data not in doc:
                elem.remove()

        # Add any new elements.
        for workpiece in doc.workpieces:
            self.add_workpiece(workpiece)
        for workstep in doc.workplan:
            self.add_workstep(workstep)

    def add_workstep(self, workstep):
        """
        Adds the workstep, but only if it does not yet exist.
        Also adds each of the WorkPieces, but only if they
        do not exist.
        """
        # Add or find the WorkStep.
        elem = cast(WorkStepElement, self.find_by_data(workstep))
        if not elem:
            # WorkStepElement should cover the entire canvas area in pixels
            elem = WorkStepElement(
                workstep,
                0,  # x_px
                0,  # y_px
                self.root.width,  # width_px
                self.root.height,  # height_px
                canvas=self,
                parent=self.root,
            )
            self.add(elem)
            workstep.changed.connect(self.on_workstep_changed)
        self.queue_draw()

        # Ensure WorkPieceOpsElements are created for each WorkPiece
        for workpiece in workstep.workpieces():
            elem.add_workpiece(workpiece)

    def set_laser_dot_visible(self, visible=True):
        self.laser_dot.set_visible(visible)
        self.queue_draw()

    def set_laser_dot_position(self, x_mm, y_mm):
        """Sets the laser dot position in real-world mm."""
        # LaserDotElement is sized to represent the dot diameter in pixels.
        # Its position should be the top-left corner of its bounding box.
        # We want the center of the dot to be at (x_px, y_px).
        x_px, y_px = self.laser_dot.mm_to_pixel(x_mm, y_mm)
        dot_width_px = self.laser_dot.width
        self.laser_dot.set_pos(
            round(x_px - dot_width_px / 2), round(y_px - dot_width_px / 2)
        )
        self.queue_draw()

    def on_workstep_changed(self, workstep, **kwargs):
        elem = self.find_by_data(workstep)
        if not elem:
            return
        elem.set_visible(workstep.visible)
        self.queue_draw()

    def add_workpiece(self, workpiece):
        """
        Adds a workpiece.
        """
        if self.workpiece_elements.find_by_data(workpiece):
            self.queue_draw()
            return
        # Get workpiece natural size and work surface size
        wp_width_nat_mm, wp_height_nat_mm = workpiece.get_default_size()
        ws_width_mm = self.width_mm
        ws_height_mm = self.height_mm

        # Determine the size to use in mm, scaling down if necessary
        width_mm = wp_width_nat_mm
        height_mm = wp_height_nat_mm

        if wp_width_nat_mm > ws_width_mm or wp_height_nat_mm > ws_height_mm:
            # Calculate scaling factor while maintaining aspect ratio
            scale_w = (
                ws_width_mm / wp_width_nat_mm if wp_width_nat_mm > 0 else 1
            )
            scale_h = (
                ws_height_mm / wp_height_nat_mm if wp_height_nat_mm > 0 else 1
            )
            scale = min(scale_w, scale_h)

            width_mm = wp_width_nat_mm * scale
            height_mm = wp_height_nat_mm * scale

        # Calculate desired position in mm (centered)
        x_mm = ws_width_mm / 2 - width_mm / 2
        y_mm = ws_height_mm / 2 - height_mm / 2
        workpiece.set_pos(x_mm, y_mm)

        # Set the workpiece's size in mm
        workpiece.set_size(width_mm, height_mm)

        # Create and add the workpiece element with pixel dimensions
        elem = WorkPieceElement(
            workpiece, canvas=self, parent=self.workpiece_elements
        )
        self.workpiece_elements.add(elem)
        self.queue_draw()

    def clear_workpieces(self):
        self.workpiece_elements.clear()
        self.queue_draw()
        self.active_element_changed.send(self, element=None)

    def clear(self):
        # Clear all children except the fixed ones
        # (workpiece_elements, laser_dot)
        children_to_remove = [
            c
            for c in self.root.children
            if c not in [self.workpiece_elements, self.laser_dot]
        ]
        for child in children_to_remove:
            child.remove()
        # Clear children of workpiece_elements
        self.workpiece_elements.clear()
        self.queue_draw()

    def find_by_type(self, thetype):
        """
        Search recursively through the root's children
        """
        return self.root.find_by_type(thetype)

    def set_workpieces_visible(self, visible=True):
        self.workpiece_elements.set_visible(visible)
        self.queue_draw()

    def do_snapshot(self, snapshot):
        # Create a Cairo context for the snapshot
        width, height = self.get_width(), self.get_height()
        bounds = Graphene.Rect().init(0, 0, width, height)
        ctx = snapshot.append_cairo(bounds)

        # Draw grid, axis, and labels
        self.axis_renderer.draw_grid(ctx)
        self.axis_renderer.draw_axes_and_labels(ctx)

        super().do_snapshot(snapshot)

    def on_key_pressed(
        self, controller, keyval: int, keycode: int, state: Gdk.ModifierType
    ):
        if keyval == Gdk.KEY_Delete:
            selected = [
                e
                for e in self.root.get_selected_data()
                if isinstance(e, WorkPiece)
            ]
            for workpiece in selected:
                for step_elem in self.find_by_type(WorkStepElement):
                    ops_elem = step_elem.find_by_data(workpiece)
                    if not ops_elem:
                        continue
                    ops_elem.remove()
                    del ops_elem  # to ensure signals disconnect
        return super().on_key_pressed(controller, keyval, keycode, state)

    def on_pan_begin(self, gesture, x, y):
        self.pan_start = (
            self.axis_renderer.pan_x_mm,
            self.axis_renderer.pan_y_mm,
        )

    def on_pan_update(self, gesture, x, y):
        # Calculate pan offset based on drag delta
        offset = gesture.get_offset()
        new_pan_x_mm = self.pan_start[0] - offset.x / self.pixels_per_mm_x
        new_pan_y_mm = self.pan_start[1] + offset.y / self.pixels_per_mm_y
        self.set_pan(new_pan_x_mm, new_pan_y_mm)

    def on_pan_end(self, gesture, x, y):
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
