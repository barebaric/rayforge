import logging
from typing import cast
from gi.repository import Graphene, Gdk  # type: ignore
from ...models.workpiece import WorkPiece
from ..canvas import Canvas, CanvasElement
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
        self.show_travel_moves = False
        self.width_mm = 100.0
        self.height_mm = 100.0
        self.pixels_per_mm_x = 0.0
        self.pixels_per_mm_y = 0.0
        self.grid_size_mm = 10.0  # in mm

        # These elements will be sized and positioned in pixels by WorkSurface
        self.workpiece_elements = CanvasElement(0, 0, 0, 0, selectable=False)
        self.root.add(self.workpiece_elements)

        # DotElement size will be set in pixels by WorkSurface
        # Initialize with zero size, size and position will be set in do_size_allocate
        self.laser_dot = DotElement(0, 0, 0, 0)
        self.root.add(self.laser_dot)

        # Initial size will be set by do_size_allocate
        # self.update() # update is called by set_size

    def _mm_to_canvas_pixel(self, x_mm, y_mm):
        """Converts real-world mm coordinates to canvas pixel coordinates."""
        # Assuming canvas origin is top-left, y-down.
        # Real-world origin is typically bottom-left, y-up.
        # Need to flip y-axis and scale.
        x_px = x_mm * self.pixels_per_mm_x
        y_px = self.root.height - (y_mm * self.pixels_per_mm_y)
        return x_px, y_px

    def _canvas_pixel_to_mm(self, x_canvas_pixel, y_canvas_pixel):
        """Converts canvas pixel coordinates to real-world mm coordinates."""
        # Assuming canvas origin is top-left, y-down.
        # Real-world origin is typically bottom-left, y-up.
        # Need to flip y-axis and scale.
        x_mm = x_canvas_pixel / self.pixels_per_mm_x if self.pixels_per_mm_x else 0
        #y_mm = (self.height_mm * self.pixels_per_mm_y - y_canvas_pixel) / self.pixels_per_mm_y if self.pixels_per_mm_y > 0 else 0
        y_mm = self.height_mm - y_canvas_pixel/self.pixels_per_mm_y
        return x_mm, y_mm

    def do_size_allocate(self, width, height, baseline):
        """Handles canvas size allocation in pixels."""
        logger.debug(f"WorkSurface.do_size_allocate: width={width}, height={height}, baseline={baseline}")
        # Check if the size has actually changed
        if width == self.root.width and height == self.root.height:
            logger.debug("WorkSurface.do_size_allocate: Size has not changed, skipping re-allocation")
            return

        # Set the root element's size directly in pixels
        self.root.set_size(width, height)

        # Update WorkSurface's internal pixel dimensions
        self.pixels_per_mm_x = width / self.width_mm if self.width_mm > 0 else 0
        self.pixels_per_mm_y = height / self.height_mm if self.height_mm > 0 else 0
        logger.debug(
            "WorkSurface.do_size_allocate: width=%s, height=%s, pixels_per_mm_x=%s, pixels_per_mm_y=%s, width_mm=%s, height_mm=%s",
            width, height, self.pixels_per_mm_x, self.pixels_per_mm_y, self.width_mm, self.height_mm
        )

        # Update children elements that need to match canvas size
        self.workpiece_elements.set_size(width, height)
        logger.debug(f"WorkSurface.do_size_allocate: width={width}, height={height}, pixels_per_mm_x={self.pixels_per_mm_x}, pixels_per_mm_y={self.pixels_per_mm_y}")

        # Update laser dot size based on new pixel dimensions and its mm radius
        dot_radius_mm = self.laser_dot.radius_mm
        dot_diameter_px = round(2 * dot_radius_mm * self.pixels_per_mm_x)
        self.laser_dot.set_size(dot_diameter_px, dot_diameter_px)

        # Re-position laser dot based on new pixel dimensions
        # Need to get current mm position first
        # Use pos_abs() to get position relative to canvas root in pixels
        current_dot_pos_px = self.laser_dot.pos_abs()
        current_dot_pos_mm = self._canvas_pixel_to_mm(*current_dot_pos_px)
        self.set_laser_dot_position(*current_dot_pos_mm) # This will convert back to new pixels

        # Allocate children based on new pixel sizes
        for elem in self.find_by_type(WorkStepElement):
            elem.set_size(width, height)
        
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
        # Convert mm position to canvas pixel position
        x_px, y_px = self._mm_to_canvas_pixel(x_mm, y_mm)

        # LaserDotElement is sized to represent the dot diameter in pixels.
        # Its position should be the top-left corner of its bounding box.
        # We want the center of the dot to be at (x_px, y_px).
        dot_width_px = self.laser_dot.width
        self.laser_dot.set_pos(x_px - dot_width_px / 2,
                               y_px - dot_width_px / 2)
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
        logger.debug("WorkSurface: do_snapshot called (actual redraw)")

        # Create a Cairo context for the snapshot
        width, height = self.get_width(), self.get_height()
        bounds = Graphene.Rect().init(0, 0, width, height)
        ctx = snapshot.append_cairo(bounds)

        self.pixels_per_mm_x = width/self.width_mm if self.width_mm > 0 else 0
        self.pixels_per_mm_y = height/self.height_mm if self.height_mm > 0 else 0
        self._draw_grid(ctx, width, height)

        super().do_snapshot(snapshot)

    def _draw_grid(self, ctx, width, height):
        """Draws the grid on the work surface."""
        if self.grid_size_mm <= 0 or self.pixels_per_mm_x <= 0 or self.pixels_per_mm_y <= 0:
            return

        grid_size_px_x = self.grid_size_mm * self.pixels_per_mm_x
        grid_size_px_y = self.grid_size_mm * self.pixels_per_mm_y

        ctx.set_source_rgba(0.5, 0.5, 0.5, 0.5)  # Gray, semi-transparent
        ctx.set_line_width(1.0)
        ctx.set_hairline(True)

        # Draw vertical lines
        x = 0
        while x <= width:
            ctx.move_to(x, 0)
            ctx.line_to(x, height)
            x += grid_size_px_x

        # Draw horizontal lines
        y = 0
        while y <= height:
            ctx.move_to(0, y)
            ctx.line_to(width, y)
            y += grid_size_px_y

        ctx.stroke()

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