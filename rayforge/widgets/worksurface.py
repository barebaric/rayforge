import math
import logging
from typing import Optional, Tuple, cast
from gi.repository import Graphene, GLib, Gdk  # type: ignore
import cairo
from ..models.ops import Ops
from ..opsencoder.cairoencoder import CairoEncoder
from ..config import config
from ..models.workpiece import WorkPiece
from ..models.workplan import WorkStep
from .canvas import Canvas, CanvasElement


logger = logging.getLogger(__name__)


def _copy_surface(source, target, width, height, clip):
    in_width, in_height = source.get_width(), source.get_height()
    scale_x = width/in_width
    scale_y = height/in_height
    ctx = cairo.Context(target)
    # Apply clipping in the target context before scaling and painting
    if clip is not None:
        clip_x, clip_y, clip_w, clip_h = clip
        ctx.rectangle(clip_x, clip_y, clip_w, clip_h)
        ctx.clip()
    ctx.scale(scale_x, scale_y)
    ctx.set_source_surface(source, 0, 0) # Set source surface at (0,0)
    ctx.paint()
    return target


class WorkPieceElement(CanvasElement):
    """
    A CanvasElement that displays a WorkPiece.

    It handles position and size updates based on the WorkPiece data,
    and uses _copy_surface to render the WorkPiece's surface.
    """
    def __init__(self, workpiece: WorkPiece, **kwargs):
        """
        Initializes a new WorkPieceElement.

        Args:
            workpiece: The WorkPiece to display.
            **kwargs: Additional keyword arguments for CanvasElement.
        """
        self.canvas: WorkSurface | None
        self.data: WorkPiece = workpiece
        self._last_pos_mm: Optional[Tuple[float, float]] = None
        self._last_size_mm: Optional[Tuple[float, float]] = None
        x_mm, y_mm = workpiece.pos or (0, 0)
        width_mm, height_mm = workpiece.size or workpiece.get_default_size()
        super().__init__(0, 0, 0, 0, data=workpiece, **kwargs)
        self._last_pos_mm = (x_mm, y_mm)
        self._last_size_mm = (width_mm, height_mm)
        workpiece.size_changed.connect(self.allocate)
        workpiece.changed.connect(self._on_workpiece_changed)

    def _on_workpiece_changed(self, workpiece: WorkPiece):
        """
        Handles changes to the WorkPiece data.

        Updates the element's position and size if they have changed
        significantly.

        Args:
            workpiece: The WorkPiece that has changed.
        """
        if not self.canvas:
            return

        # Get the new position and size in mm.
        x_mm, y_mm = workpiece.pos or (0, 0)
        width_mm, height_mm = workpiece.size or (0, 0)

        # Convert the mm values to pixel values.
        new_x, new_y = self.canvas._mm_to_canvas_pixel(x_mm, y_mm + height_mm)
        new_width = round(width_mm * self.canvas.pixels_per_mm_x)
        new_height = round(height_mm * self.canvas.pixels_per_mm_y)

        # Check if the position or size has changed significantly.
        if (
            abs(new_x - self.x) >= 1
            or abs(new_y - self.y) >= 1
            or abs(new_width - self.width) >= 1
            or abs(new_height - self.height) >= 1
        ):
            # Update the element's position and size.
            self.x, self.y = new_x, new_y
            self.width, self.height = new_width, new_height

            # Update the last known position and size in mm.
            self._last_pos_mm = (x_mm, y_mm)
            self._last_size_mm = (width_mm, height_mm)

            # Allocate the element and mark it as dirty.
            super().allocate()
            self.mark_dirty()
            self.canvas.queue_draw()


    def _update_workpiece(self):
        """
        Updates the WorkPiece data with the element's current position and size.
        """
        if not self.canvas:
            return

        # Get the element's position and size in pixels.
        x, y, width, height = self.rect_abs()

        # Convert the pixel values to mm values.
        x_mm, y_mm = self.canvas._canvas_pixel_to_mm(x, y + height)
        width_mm = width / self.canvas.pixels_per_mm_x
        height_mm = height / self.canvas.pixels_per_mm_y

        # Update the WorkPiece's position if it has changed.
        if self._last_pos_mm is None or (x_mm, y_mm) != self._last_pos_mm:
            self.data.set_pos(x_mm, y_mm)
            self._last_pos_mm = (x_mm, y_mm)

        # Update the WorkPiece's size if it has changed.
        if (
            self._last_size_mm is None
            or (width_mm, height_mm) != self._last_size_mm
        ):
            self.data.set_size(width_mm, height_mm)
            self._last_size_mm = (width_mm, height_mm)


    def allocate(self, force: bool = False):
        """
        Allocates the element's position and size based on the WorkPiece data.

        Args:
            force: Whether to force allocation, even if the position and size
                have not changed.
        """
        if not self.canvas:
            return

        # Get the position and size in mm from the WorkPiece.
        x_mm, y_mm = self.data.pos or (0, 0)
        width_mm, height_mm = self.data.size or self.data.get_default_size()

        # Convert the mm values to pixel values.
        new_x, new_y = self.canvas._mm_to_canvas_pixel(x_mm, y_mm + height_mm)
        new_width = round(width_mm * self.canvas.pixels_per_mm_x)
        new_height = round(height_mm * self.canvas.pixels_per_mm_y)

        # Check if the position or size has changed significantly.
        if (
            force
            or abs(new_x - self.x) >= 1
            or abs(new_y - self.y) >= 1
            or abs(new_width - self.width) >= 1
            or abs(new_height - self.height) >= 1
        ):
            # Update the element's position and size.
            self.x, self.y = new_x, new_y
            self.width, self.height = new_width, new_height

            # Update the last known position and size in mm.
            self._last_pos_mm = (x_mm, y_mm)
            self._last_size_mm = (width_mm, height_mm)

            # Allocate the element and mark it as dirty.
            super().allocate(force)
            self.mark_dirty()
            self.canvas.queue_draw()

    def render(
        self,
        clip: tuple[float, float, float, float] | None = None,
        force: bool = False,
    ):
        """
        Renders the WorkPiece element to the canvas.

        Args:
            clip: The clipping rectangle, or None for no clipping.
            force: Whether to force rendering, even if the element is not dirty.
        """
        if not self.dirty and not force:
            return
        if not self.canvas or not self.parent or self.surface is None:
            return
        surface, changed = self.data.render(
            self.canvas.pixels_per_mm_x,
            self.canvas.pixels_per_mm_y,
            (self.width, self.height),
        )
        if not changed or surface is None:
            return
        self.clear_surface(clip or self.rect())
        self.surface = _copy_surface(
            surface,
            self.surface,
            self.width,
            self.height,
            clip or (0, 0, self.width, self.height),
        )
        self.dirty = False

    def set_pos(self, x: int, y: int):
        """
        Sets the position of the element in pixels.

        Args:
            x: The new x-coordinate in pixels.
            y: The new y-coordinate in pixels.
        """
        super().set_pos(x, y)
        self._update_workpiece()

    def set_size(self, width: int, height: int):
        """
        Sets the size of the element in pixels.

        Args:
            width: The new width in pixels.
            height: The new height in pixels.
        """
        super().set_size(width, height)
        self._update_workpiece()



class WorkPieceOpsElement(CanvasElement):
    """Displays the generated Ops for a single WorkPiece."""
    def __init__(self, workpiece: WorkPiece, **kwargs):
        """
        Initializes a WorkPieceOpsElement.

        Args:
            workpiece: The WorkPiece data object.
            **kwargs: Additional keyword arguments for CanvasElement.
        """
        if not workpiece.size:
            raise AttributeError(f"attempt to add workpiece {workpiece.name} with no size")
        super().__init__(0,
                         0,
                         0,
                         0,
                         data=workpiece,
                         selectable=False,
                         **kwargs)
        self._accumulated_ops = Ops()
        workpiece.changed.connect(self._on_workpiece_changed)

    def allocate(self, force: bool = False):
        """Updates the element's position and size based on the workpiece."""
        if not self.canvas:
            return

        # Even though allocate() does not require the position, we update
        # it here anyway to do it as early as possible. We need to update
        # because the pixels_per_mm in the canvas may have changed, requiring
        # a re-caclulation of the positions in pixel.
        x_mm, y_mm = self.data.pos or (0, 0)
        width_mm, height_mm = self.data.size or (0, 0)
        self.x, self.y = self.canvas._mm_to_canvas_pixel(x_mm, y_mm+height_mm)

        # Calculate the size of the surface.
        width_mm, height_mm = self.data.size or (0, 0)
        pixels_per_mm_x = self.canvas.pixels_per_mm_x
        pixels_per_mm_y = self.canvas.pixels_per_mm_y
        width_px = round(width_mm * pixels_per_mm_x)
        height_px = round(height_mm * pixels_per_mm_y)
        self.width, self.height = width_px, height_px

        super().allocate(force)

    def _on_workpiece_changed(self, workpiece: WorkPiece):
        # Workpiece position and/or size in mm changed.
        logger.debug(f"WorkPieceOpsElement._on_workpiece_changed: Workpiece {workpiece.name} changed. Requesting redraw. pos={workpiece.pos}, size={workpiece.size}")
        if not self.canvas:
            return
        self.allocate()
        self.mark_dirty()
        self.canvas.queue_draw()

    def clear_ops(self):
        """Clears the accumulated operations and the drawing surface."""
        self._accumulated_ops = Ops()
        self.clear_surface()
        self.mark_dirty()

    def add_ops(self, ops_chunk: Ops):
        """Adds a chunk of operations to the accumulated total."""
        if not ops_chunk:
            return
        self._accumulated_ops += ops_chunk
        self.mark_dirty()

    def render(self, clip: tuple[float, float, float, float] | None = None, force: bool = False):
        """Renders the accumulated Ops to the element's surface."""
        logger.debug(f"WorkPieceOpsElement.render: Workpiece {self.data.name}. clip={clip}, force={force}")
        if not self.dirty and not force:
            return
        if not self.canvas or not self.parent or self.surface is None:
            return

        # Clear the surface.
        clip = clip or self.rect()
        self.clear_surface(clip)

        if not self._accumulated_ops:
            return

        # Get pixels_per_mm from the WorkSurface (self.canvas)
        pixels_per_mm = self.canvas.pixels_per_mm_x, self.canvas.pixels_per_mm_y

        encoder = CairoEncoder()
        show_travel = self.canvas.show_travel_moves if self.canvas else False
        encoder.encode(self._accumulated_ops,
                       config.machine,
                       self.surface,
                       pixels_per_mm,
                       show_travel_moves=show_travel)


class WorkStepElement(CanvasElement):
    """
    WorkStepElements display the result of a WorkStep on the
    WorkSurface. The output represents the laser path.
    """
    def __init__(self, workstep, x, y, width, height, **kwargs):
        """
        Initializes a WorkStepElement with pixel dimensions.

        Args:
            workstep: The WorkStep data object.
            x: The x-coordinate (pixel) relative to the parent.
            y: The y-coordinate (pixel) relative to the parent.
            width: The width (pixel).
            height: The height (pixel).
            **kwargs: Additional keyword arguments for CanvasElement.
        """
        super().__init__(x,
                         y,
                         width,
                         height,
                         data=workstep,
                         selectable=False,
                         **kwargs)
        workstep.changed.connect(self._on_workstep_changed)
        # Connect to the actual signals from WorkStep's async pipeline
        workstep.ops_generation_starting.connect(
            self._on_ops_generation_starting
        )
        # Note: There is no explicit 'cleared' signal in the async pipeline,
        # starting implies clearing for the UI representation.
        workstep.ops_chunk_available.connect(
            self._on_ops_chunk_available
        )
        workstep.ops_generation_finished.connect(
            self._on_ops_generation_finished
        )
        # Workpiece elements are added dynamically when ops chunks arrive

    def add_workpiece(self, workpiece) -> WorkPieceOpsElement:
        """
        Adds a WorkPieceOpsElement for the given workpiece if it doesn't exist.
        Returns the existing or newly created element.
        """
        elem = self.find_by_data(workpiece)
        if elem:
            elem.mark_dirty()
            return cast(WorkPieceOpsElement, elem)

        elem = WorkPieceOpsElement(workpiece,
                                   canvas=self.canvas,
                                   parent=self)
        self.add(elem)
        return elem

    def _on_workstep_changed(self, step: WorkStep):
        # This signal is for changes to the WorkStep itself (e.g., visibility)
        # not changes to its workpieces or ops.
        # Workpiece additions/removals are handled by the ops generation signals.
        # We just need to update visibility and redraw.
        assert self.canvas, "Received ops_start, but element was not added to canvas"
        self.set_visible(step.visible)
        if self.canvas:
            self.canvas.queue_draw()

    def _find_or_add_workpiece_elem(self, workpiece: WorkPiece) -> WorkPieceOpsElement:
        """Finds the element for a workpiece, creating if necessary."""
        elem = cast(Optional[WorkPieceOpsElement], self.find_by_data(workpiece))
        if not elem:
            logger.debug(f"Adding workpiece to step: {workpiece.name}")
            elem = self.add_workpiece(workpiece)
        return elem

    def _on_ops_generation_starting(self,
                                    sender: WorkStep,
                                    workpiece: WorkPiece):
        """Called before ops generation starts for a workpiece."""
        logger.debug(
            f"WorkStepElem '{sender.name}': Received ops_generation_starting "
            f"for {workpiece.name}"
        )
        assert self.canvas, "Received ops_start, but element was not added to canvas"
        elem = self._find_or_add_workpiece_elem(workpiece)
        elem.clear_ops()
        GLib.idle_add(self.canvas.queue_draw)

    def _on_ops_chunk_available(self, sender: WorkStep, workpiece: WorkPiece,
                                chunk: Ops):
        """Called when a chunk of ops is available for a workpiece."""
        logger.debug(
            f"WorkStepElem '{sender.name}': Received ops_chunk_available for "
            f"{workpiece.name} (chunk size: {len(chunk)}, pos={workpiece.pos})"
        )
        assert self.canvas, "Received update, but element was not added to canvas"
        elem = self._find_or_add_workpiece_elem(workpiece)
        elem.add_ops(chunk)
        GLib.idle_add(self.canvas.queue_draw)

    def _on_ops_generation_finished(self, sender: WorkStep, workpiece: WorkPiece):
        """Called when ops generation is finished for a workpiece."""
        # Final redraw is triggered by the last _on_ops_chunk_available call's
        # queue_draw. No extra action needed here unless we add UI
        # indicators for processing state (e.g., hide a spinner).
        assert self.canvas, "Received ops_finished, but element was not added to canvas"
        GLib.idle_add(self.canvas.queue_draw)


class LaserDotElement(CanvasElement):
    """
    Draws a simple red dot.
    """
    def __init__(self, x, y, width, height, **kwargs):
        """
        Initializes a LaserDotElement with pixel dimensions.

        Args:
            x: The x-coordinate (pixel) relative to the parent.
            y: The y-coordinate (pixel) relative to the parent.
            width: The width (pixel).
            height: The height (pixel).
            **kwargs: Additional keyword arguments for CanvasElement.
        """
        # Laser dot is always a circle, so width and height should be equal.
        # We store the radius in mm for rendering purposes.
        self.radius_mm = 1.0 # Default radius in mm
        super().__init__(x,
                         y,
                         width,
                         height,
                         visible=True,
                         selectable=False,
                         **kwargs)

    def render(self, clip: tuple[float, float, float, float] | None = None, force: bool = False):
        """Renders the laser dot to the element's surface."""
        if not self.dirty and not force:
            return
        if not self.canvas or not self.parent or self.surface is None:
            return

        # Clear the surface.
        clip = clip or self.rect()
        self.clear_surface(clip)

        # Prepare the context.
        ctx = cairo.Context(self.surface)
        ctx.set_hairline(True)
        ctx.set_source_rgb(.9, 0, 0)

        # Calculate radius in pixels based on the stored mm radius
        pixels_per_mm_x = self.canvas.pixels_per_mm_x
        radius_px = self.radius_mm * pixels_per_mm_x

        # Draw the circle centered within the element's pixel bounds
        center_x = self.width / 2
        center_y = self.height / 2
        ctx.arc(center_x, center_y, radius_px, 0., 2*math.pi)
        ctx.fill()


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

        # LaserDotElement size will be set in pixels by WorkSurface
        # Initialize with zero size, size and position will be set in do_size_allocate
        self.laser_dot = LaserDotElement(0, 0, 0, 0)
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
