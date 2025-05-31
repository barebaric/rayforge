import logging
from typing import cast
from gi.repository import Graphene, Gdk, Gtk  # type: ignore
from ...models.workpiece import WorkPiece
from ..canvas import Canvas, CanvasElement
from .axis import AxisRenderer
from .dotelem import DotElement
from .workstepelem import WorkStepElement
from .workpieceelem import WorkPieceElement


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
        self.pan_x_mm = 0.0
        self.pan_y_mm = 0.0
        self.show_travel_moves = False
        self.width_mm = 100.0
        self.height_mm = 100.0
        self.pixels_per_mm_x = 0.0
        self.pixels_per_mm_y = 0.0
        self.grid_size_mm = 10.0  # in mm
        self.font_size = 10  # Hardcoded font size

        self.axis_renderer = AxisRenderer(
            font_size=self.font_size,
            width_mm=self.width_mm,
            height_mm=self.height_mm,
            pan_x_mm=self.pan_x_mm,
            pan_y_mm=self.pan_y_mm,
            zoom_level=self.zoom_level,
        )
        self.root.background = 0.9, 0.9, 0.9, 0.1  # light gray background

        # These elements will be sized and positioned in pixels by WorkSurface
        self.workpiece_elements = CanvasElement(0, 0, 0, 0, selectable=False)
        self.root.add(self.workpiece_elements)

        # DotElement size will be set in pixels by WorkSurface
        # Initialize with zero size, size and position will be set in do_size_allocate
        self.laser_dot = DotElement(0, 0, 0, 0)
        self.root.add(self.laser_dot)

        # Add scroll event controller for zoom
        self.scroll_controller = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags.VERTICAL)
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

        # Initial size will be set by do_size_allocate
        # self.update() # update is called by set_size

    def on_scroll(self, controller, dx, dy):
        """Handles the scroll event for zoom."""
        zoom_speed = 0.00    # Zoom disabled; does not work yet
        if dy > 0:
            self.zoom_level -= zoom_speed
        else:
            self.zoom_level += zoom_speed
        self.zoom_level = max(0.5, self.zoom_level)  # Prevent zoom level from becoming too small

        # Update AxisRenderer with the new zoom level
        self.axis_renderer.set_zoom(self.zoom_level)

        self.do_size_allocate(self.get_width(), self.get_height(), 0)
        self.queue_draw()

    def do_size_allocate(self, width, height, baseline):
        """Handles canvas size allocation in pixels."""
        #logger.debug(f"WorkSurface.do_size_allocate: width={width}, height={height}, baseline={baseline}")
        # Check if the size has actually changed
        if width == self.root.width and height == self.root.height:
            logger.debug("WorkSurface.do_size_allocate: Size has not changed, skipping re-allocation")
            return

        # Calculate grid bounds using AxisRenderer
        self.axis_renderer.set_width_mm(self.width_mm)
        self.axis_renderer.set_height_mm(self.height_mm)
        self.axis_renderer.set_pan_x_mm(self.pan_x_mm)
        self.axis_renderer.set_pan_y_mm(self.pan_y_mm)
        self.axis_renderer.set_zoom(self.zoom_level)
        origin_x, origin_y, max_x, max_y = self.axis_renderer.get_grid_bounds(width, height)
        axis_width = self.axis_renderer.get_y_axis_width()
        axis_height = self.axis_renderer.get_x_axis_height()

        # Calculate content area based on grid bounds
        content_width, content_height = self.axis_renderer.get_content_size(width, height)

        # Update WorkSurface's internal pixel dimensions based on content area
        self.pixels_per_mm_x = content_width / self.width_mm if self.width_mm > 0 else 0
        self.pixels_per_mm_y = content_height / self.height_mm if self.height_mm > 0 else 0

        # Set the root element's size directly in pixels
        self.root.set_pos(2*axis_width - origin_x, height - origin_y - axis_height)
        self.root.set_size(content_width, content_height)

        # Update child elements that need to match canvas size
        self.workpiece_elements.set_size(content_width, content_height)

        # Update laser dot size based on new pixel dimensions and its mm radius
        dot_radius_mm = self.laser_dot.radius_mm
        dot_diameter_px = round(2 * dot_radius_mm * self.pixels_per_mm_x)
        self.laser_dot.set_size(dot_diameter_px, dot_diameter_px)

        # Re-position laser dot based on new pixel dimensions
        # Need to get current mm position first
        # Use pos_abs() to get position relative to canvas root in pixels
        current_dot_pos_px = self.laser_dot.pos_abs()
        current_dot_pos_mm = self.laser_dot.pixel_to_mm(*current_dot_pos_px)
        self.set_laser_dot_position(*current_dot_pos_mm)  # This will convert back to new pixels

        # Allocate children based on new pixel sizes
        for elem in self.find_by_type(WorkStepElement):
            elem.set_size(content_width, content_height)

        self.root.mark_dirty(recursive=True)
        self.root.allocate()
        self.queue_draw()

    def set_show_travel_moves(self, show: bool):
        """Sets whether to display travel moves and triggers a redraw."""
        if self.show_travel_moves != show:
            self.show_travel_moves = show
            # Mark elements dirty that depend on this setting
            for elem in self.find_by_type(WorkStepElement):
                elem.mark_dirty()
            self.queue_draw()

    def set_size(self, width_mm, height_mm):
        """Sets the real-world size of the work surface in mm."""
        self.width_mm = width_mm
        self.height_mm = height_mm
        # The actual pixel size and pixels_per_mm will be calculated in do_size_allocate
        # when the canvas widget is allocated.
        self.update()

    def update(self):
        logger.debug("WorkSurface.update called")
        """Updates internal state and triggers redraw."""
        # This method is now primarily for triggering redraws or updates
        # that don't depend on pixel allocation.
        # Aspect ratio calculation might still be useful here.
        self.aspect_ratio = self.width_mm / self.height_mm if self.height_mm > 0 else 1.0
        self.queue_draw()

    def update_from_doc(self, doc):
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
            elem = WorkStepElement(workstep,
                                   0, # x_px
                                   0, # y_px
                                   self.root.width, # width_px
                                   self.root.height, # height_px
                                   canvas=self,
                                   parent=self.root)
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
        self.laser_dot.set_pos(round(x_px - dot_width_px / 2),
                               round(y_px - dot_width_px / 2))
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
        x_mm = ws_width_mm/2 - width_mm/2
        y_mm = ws_height_mm/2 - height_mm/2
        workpiece.set_pos(x_mm, y_mm)

        # Set the workpiece's size in mm
        workpiece.set_size(width_mm, height_mm)

        # Create and add the workpiece element with pixel dimensions
        elem = WorkPieceElement(workpiece,
                                canvas=self,
                                parent=self.workpiece_elements)
        self.workpiece_elements.add(elem)
        self.queue_draw()

    def clear_workpieces(self):
        self.workpiece_elements.clear()
        self.queue_draw()

    def clear(self):
        # Clear all children except the fixed ones (workpiece_elements, laser_dot)
        children_to_remove = [
            c for c in self.root.children
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

        # Draw grid
        self.axis_renderer.draw_grid(
            ctx,
            width,
            height,
        )

        # Draw axes and labels
        self.axis_renderer.draw_axes_and_labels(
            ctx,
            width,
            height,
        )

        super().do_snapshot(snapshot)

    def on_key_pressed(self, controller, keyval: int, keycode: int, state: Gdk.ModifierType):
        if keyval == Gdk.KEY_Delete:
            selected = [e for e in self.root.get_selected_data() if isinstance(e, WorkPiece)]
            for workpiece in selected:
                for step_elem in self.find_by_type(WorkStepElement):
                    ops_elem = step_elem.find_by_data(workpiece)
                    if not ops_elem:
                        continue
                    ops_elem.remove()
                    del ops_elem   # to ensure signals disconnect
        return super().on_key_pressed(controller, keyval, keycode, state)

    def on_pan_begin(self, gesture, x, y):
        self.pan_start = self.pan_x_mm, self.pan_y_mm
        logger.debug(f"on_pan_begin: x={x}, y={y}")

    def on_pan_update(self, gesture, x, y):
        # Calculate pan offset based on drag delta
        offset = gesture.get_offset()
        self.pan_x_mm = self.pan_start[0] - offset.x/self.pixels_per_mm_x
        self.pan_y_mm = self.pan_start[1] + offset.y/self.pixels_per_mm_y
        self.do_size_allocate(self.get_width(), self.get_height(), 0)
        self.queue_draw()

    def on_pan_end(self, gesture, x, y):
        pass