# flake8: noqa: E402
import gi
import logging
import gettext
import math
from pathlib import Path

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk
import cairo
from typing import Optional, Dict, Tuple, List

base_path = Path(__file__).parent
gettext.install("canvas", base_path / "rayforge" / "locale")
logging.basicConfig(level=logging.DEBUG)


from rayforge.workbench.canvas import Canvas, CanvasElement, ShrinkWrapGroup
from rayforge.core.matrix import Matrix


class ExampleElement(CanvasElement):
    """
    A custom canvas element that draws a triangle to clearly show its
    orientation, useful for debugging transformations.

    This element also demonstrates custom drag behavior by snapping to a grid
    if `snap_grid_size` is provided.
    """

    def __init__(
        self, *args, snap_grid_size: Optional[float] = None, **kwargs
    ):
        """
        Initializes the ExampleElement.

        Args:
            snap_grid_size: If set, the element will snap to a grid of
                            this size when dragged. `draggable` must also
                            be set to True for this to take effect.
        """
        super().__init__(*args, **kwargs)
        self.snap_grid_size = snap_grid_size

    def _draw_content(self, ctx: cairo.Context, width: float, height: float):
        """Draws the orienting shape (a triangle)."""
        ctx.set_source_rgba(0.1, 0.1, 0.1, 0.7)
        ctx.move_to(width / 2, height * 0.1)  # Top point
        ctx.line_to(width * 0.1, height * 0.9)  # Bottom-left
        ctx.line_to(width * 0.9, height * 0.9)  # Bottom-right
        ctx.close_path()
        ctx.fill()

    def draw(self, ctx: cairo.Context):
        """Overrides drawing for unbuffered elements."""
        ctx.save()
        cairo_content_matrix = cairo.Matrix(
            *self.content_transform.for_cairo()
        )
        ctx.transform(cairo_content_matrix)

        if not self.buffered:
            ctx.set_source_rgba(*self.background)
            ctx.rectangle(0, 0, self.width, self.height)
            ctx.fill()
            self._draw_content(ctx, self.width, self.height)
        else:
            super().draw(ctx)

        ctx.restore()

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """Overrides surface rendering for buffered elements."""
        surface = super().render_to_surface(width, height)
        if surface:
            ctx = cairo.Context(surface)
            self._draw_content(ctx, width, height)
        return surface

    def handle_drag_move(
        self, world_dx: float, world_dy: float
    ) -> Tuple[float, float]:
        """Overrides the drag behavior to implement grid snapping."""
        # If no grid is set, or if the element is not on a canvas,
        # perform default (unconstrained) dragging.
        if self.snap_grid_size is None or not self.canvas:
            return super().handle_drag_move(world_dx, world_dy)

        # The canvas stores the element's state at the start of the drag.
        # This is crucial for calculating the snap from a consistent origin.
        initial_transform = self.canvas._initial_world_transform
        if initial_transform is None:
            return world_dx, world_dy  # Should not happen during a drag

        # 1. Get the initial world position of the element's origin.
        initial_x, initial_y = initial_transform.get_translation()

        # 2. Calculate the proposed new position in world coordinates.
        proposed_x = initial_x + world_dx
        proposed_y = initial_y + world_dy

        # 3. Snap the proposed position to the grid.
        grid = self.snap_grid_size
        snapped_x = round(proposed_x / grid) * grid
        snapped_y = round(proposed_y / grid) * grid

        # 4. Calculate the new, constrained delta from the initial position.
        constrained_dx = snapped_x - initial_x
        constrained_dy = snapped_y - initial_y

        return constrained_dx, constrained_dy


class LShapeElement(CanvasElement):
    """
    A custom element that draws a non-symmetric "L" shape to test
    pixel perfect hit detection.
    """

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        if width <= 0 or height <= 0:
            return None
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        padding = min(width, height) * 0.1
        thickness = min(width, height) * 0.3
        ctx.set_source_rgba(0.8, 0.2, 0.8, 1.0)
        ctx.move_to(padding, padding)
        ctx.line_to(padding, height - padding)
        ctx.line_to(width - padding, height - padding)
        ctx.line_to(width - padding, height - padding - thickness)
        ctx.line_to(padding + thickness, height - padding - thickness)
        ctx.line_to(padding + thickness, padding)
        ctx.close_path()
        ctx.fill()
        return surface


class EditableElement(CanvasElement):
    """
    A custom canvas element that can be double-clicked to enter an
    "edit mode" where its vertices can be moved.
    """

    vertices: List[List[float]]

    def __init__(self, x, y, width, height, **kwargs):
        # Ensure is_editable is set to True
        super().__init__(x, y, width, height, is_editable=True, **kwargs)
        self.original_background = self.background
        self._was_buffered = self.buffered

        # Define some editable points in local coordinates
        self.vertices = [
            [width * 0.1, height * 0.1],  # Top-left
            [width * 0.9, height * 0.1],  # Top-right
            [width * 0.9, height * 0.9],  # Bottom-right
            [width * 0.1, height * 0.9],  # Bottom-left
        ]
        self._active_vertex_idx: Optional[int] = None
        self._initial_vertex_pos: Optional[List[float]] = None

    def _draw_content(self, ctx: cairo.Context, width: float, height: float):
        """Draws a polygon connecting the vertices."""
        if not self.vertices:
            return
        ctx.set_source_rgba(0.9, 0.9, 0.2, 0.7)  # Yellowish
        ctx.move_to(*self.vertices[0])
        for v in self.vertices[1:]:
            ctx.line_to(*v)
        ctx.close_path()
        ctx.fill()

    def draw(self, ctx: cairo.Context):
        """Overrides drawing to handle both buffered and unbuffered cases."""
        ctx.save()
        cairo_content_matrix = cairo.Matrix(
            *self.content_transform.for_cairo()
        )
        ctx.transform(cairo_content_matrix)

        if not self.buffered or not self.surface:
            ctx.set_source_rgba(*self.background)
            ctx.rectangle(0, 0, self.width, self.height)
            ctx.fill()
            self._draw_content(ctx, self.width, self.height)
        else:
            super().draw(ctx)
        ctx.restore()

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """Overrides surface rendering for buffered elements."""
        surface = super().render_to_surface(width, height)
        if surface:
            ctx = cairo.Context(surface)
            # Scale the context so the vector drawing fills the surface,
            # regardless of the buffer's pixel dimensions. This ensures
            # the drawing is sharp at any zoom level.
            if self.width > 0 and self.height > 0:
                ctx.scale(width / self.width, height / self.height)
            self._draw_content(ctx, self.width, self.height)
        return surface

    def on_edit_mode_enter(self):
        logging.info("EditableElement entered edit mode.")
        self.background = (0.2, 0.2, 0.4, 1.0)  # Dark blue in edit mode
        # Store original buffered state and temporarily disable it for smooth
        # interactive drawing.
        self._was_buffered = self.buffered
        self.buffered = False
        self.trigger_update()

    def on_edit_mode_leave(self):
        logging.info("EditableElement left edit mode.")
        self.background = self.original_background
        self.buffered = self._was_buffered
        self._active_vertex_idx = None
        self._initial_vertex_pos = None
        # Clear the old surface to prevent flickering with the stale
        # content. This forces a direct draw for the first frame after
        # leaving edit mode, while the new buffered surface is generated
        # in the background.
        self.surface = None
        self.trigger_update()

    def draw_edit_overlay(self, ctx: cairo.Context):
        """Draws circular handles for each vertex in screen space."""
        if not self.canvas:
            return

        screen_transform = (
            self.canvas.view_transform @ self.get_world_transform()
        )

        ctx.save()
        ctx.set_line_width(2.0)
        handle_radius = 8.0

        for i, vertex in enumerate(self.vertices):
            sx, sy = screen_transform.transform_point((vertex[0], vertex[1]))

            if i == self._active_vertex_idx:
                ctx.set_source_rgba(1.0, 0.5, 0.0, 0.9)  # Orange for active
            else:
                ctx.set_source_rgba(0.0, 0.8, 1.0, 0.8)  # Cyan for inactive

            ctx.arc(sx, sy, handle_radius, 0, 2 * math.pi)
            ctx.fill_preserve()
            ctx.set_source_rgba(0.1, 0.1, 0.1, 1.0)
            ctx.stroke()
        ctx.restore()

    def _get_hit_vertex(self, world_x: float, world_y: float) -> Optional[int]:
        """Checks if a world coordinate point hits any vertex handle."""
        if not self.canvas:
            return None

        try:
            inv_world = self.get_world_transform().invert()
        except Exception:
            return None

        local_x, local_y = inv_world.transform_point((world_x, world_y))

        screen_transform = (
            self.canvas.view_transform @ self.get_world_transform()
        )
        scale_x, scale_y = screen_transform.get_abs_scale()

        hit_radius_local_x = 8.0 / scale_x if scale_x > 1e-6 else float("inf")
        hit_radius_local_y = 8.0 / scale_y if scale_y > 1e-6 else float("inf")

        for i, vertex in enumerate(self.vertices):
            dx = local_x - vertex[0]
            dy = local_y - vertex[1]
            # Simple bounding box check is sufficient and faster
            if abs(dx) < hit_radius_local_x and abs(dy) < hit_radius_local_y:
                return i
        return None

    def handle_edit_press(self, world_x: float, world_y: float) -> bool:
        hit_idx = self._get_hit_vertex(world_x, world_y)
        if hit_idx is not None:
            self._active_vertex_idx = hit_idx
            self._initial_vertex_pos = self.vertices[hit_idx][:]
            logging.debug(f"Editing vertex {hit_idx}")
            return True
        return False

    def handle_edit_drag(self, world_dx: float, world_dy: float):
        if (
            self._active_vertex_idx is None
            or self._initial_vertex_pos is None
            or not self.canvas
        ):
            return

        try:
            world_tf_no_trans = (
                self.get_world_transform().without_translation()
            )
            inv_rot_scale = world_tf_no_trans.invert()
            local_dx, local_dy = inv_rot_scale.transform_vector(
                (world_dx, world_dy)
            )
        except Exception:
            return

        self.vertices[self._active_vertex_idx][0] = (
            self._initial_vertex_pos[0] + local_dx
        )
        self.vertices[self._active_vertex_idx][1] = (
            self._initial_vertex_pos[1] + local_dy
        )
        # Since we are not buffered in edit mode, we must queue a draw
        # to see the changes live.
        if self.canvas:
            self.canvas.queue_draw()

    def handle_edit_release(self, world_x: float, world_y: float):
        self._active_vertex_idx = None
        self._initial_vertex_pos = None
        logging.debug("Finished editing vertex.")


class CanvasApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.CanvasApp")
        self.mouse_pos: Dict[Gtk.Widget, Tuple[float, float]] = {}
        self.initial_pan_transforms: Dict[str, Matrix] = {}

    def do_activate(self):
        win = Gtk.ApplicationWindow(application=self)
        win.set_default_size(1650, 850)
        win.set_title(
            "Canvas Test | Left: Normal Y-Down | Right: Flipped Y-Up"
        )

        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        win.set_child(main_box)

        canvas_size = 800

        # Canvas 1: Normal Coordinates (Y-axis points down)
        canvas_normal = Canvas()
        canvas_normal.set_size_request(canvas_size, canvas_size)
        main_box.append(canvas_normal)

        # Canvas 2: Flipped Coordinates (Y-axis points up)
        canvas_flipped = Canvas()
        canvas_flipped.set_size_request(canvas_size, canvas_size)

        m_scale = Matrix.scale(1, -1)
        m_trans = Matrix.translation(0, canvas_size)
        canvas_flipped.view_transform = m_trans @ m_scale
        main_box.append(canvas_flipped)

        populate_canvas(canvas_normal)
        populate_canvas(canvas_flipped)

        # Add event controllers for zoom and pan.
        for canvas in [canvas_normal, canvas_flipped]:
            # This separate motion controller is only for the app's use (zoom).
            # It will not conflict with the Canvas's internal one because the
            # ambiguity in the drag gestures has been fixed.
            motion = Gtk.EventControllerMotion.new()
            motion.connect("motion", self.on_motion)
            canvas.add_controller(motion)

            scroll = Gtk.EventControllerScroll.new(
                flags=Gtk.EventControllerScrollFlags.BOTH_AXES
            )
            scroll.connect(
                "scroll", self.on_scroll, canvas_normal, canvas_flipped
            )
            canvas.add_controller(scroll)

            drag = Gtk.GestureDrag.new()
            drag.set_button(Gdk.BUTTON_MIDDLE)
            drag.connect(
                "drag-begin", self.on_pan_begin, canvas_normal, canvas_flipped
            )
            drag.connect(
                "drag-update",
                self.on_pan_update,
                canvas_normal,
                canvas_flipped,
            )
            drag.connect("drag-end", self.on_pan_end)
            canvas.add_controller(drag)

        win.present()

    def on_motion(self, controller, x, y):
        """Stores the current mouse position for the widget."""
        canvas = controller.get_widget()
        self.mouse_pos[canvas] = (x, y)
        return Gdk.EVENT_PROPAGATE

    def on_scroll(self, controller, dx, dy, c_norm, c_flip):
        """Handles zooming on both canvases simultaneously, centered on the mouse."""
        zoom_factor = 1.1 if dy < 0 else 1 / 1.1

        canvas = controller.get_widget()
        if canvas in self.mouse_pos:
            x, y = self.mouse_pos[canvas]
        else:
            x, y = canvas.get_width() / 2, canvas.get_height() / 2

        translate_to_origin = Matrix.translation(-x, -y)
        scale = Matrix.scale(zoom_factor, zoom_factor)
        translate_back = Matrix.translation(x, y)
        zoom_matrix = translate_back @ scale @ translate_to_origin

        c_norm.view_transform = zoom_matrix @ c_norm.view_transform
        c_flip.view_transform = zoom_matrix @ c_flip.view_transform

        for c in [c_norm, c_flip]:
            for elem in c.root.get_all_children_recursive():
                if elem.buffered:
                    elem.trigger_update()
            c.queue_draw()

    def on_pan_begin(self, gesture, x, y, c_norm, c_flip):
        """Stores the initial state before a pan starts."""
        self.initial_pan_transforms["normal"] = c_norm.view_transform.copy()
        self.initial_pan_transforms["flipped"] = c_flip.view_transform.copy()

    def on_pan_update(self, gesture, offset_x, offset_y, c_norm, c_flip):
        """Updates both canvases' positions during a pan."""
        if "normal" not in self.initial_pan_transforms:
            return

        pan_delta_matrix = Matrix.translation(offset_x, offset_y)
        c_norm.view_transform = (
            pan_delta_matrix @ self.initial_pan_transforms["normal"]
        )
        c_flip.view_transform = (
            pan_delta_matrix @ self.initial_pan_transforms["flipped"]
        )

        c_norm.queue_draw()
        c_flip.queue_draw()

    def on_pan_end(self, gesture, offset_x, offset_y):
        """Clears the initial pan state."""
        self.initial_pan_transforms.clear()


def populate_canvas(canvas: Canvas):
    """Helper function to add the same elements to a canvas."""
    # This element demonstrates the custom drag handler for grid snapping.
    elem = ExampleElement(
        100,
        120,
        100,
        100,
        background=(0.5, 1, 0.5, 1),
        draggable=True,  # Make the element draggable
        snap_grid_size=50.0,  # Set the snap grid size (in world units)
    )
    canvas.add(elem)
    group = CanvasElement(250, 250, 400, 350, background=(0, 1, 1, 0.2))
    group.add(ExampleElement(20, 20, 100, 80, background=(0.7, 0.7, 1, 1)))
    group.add(
        ExampleElement(
            150, 40, 150, 150, background=(0.7, 1, 0.7, 1), buffered=True
        )
    )
    group.add(ExampleElement(50, 180, 120, 120, background=(1, 0.7, 1, 1)))
    canvas.add(group)
    l_shape = LShapeElement(
        500,
        80,
        150,
        150,
        background=(0, 0, 0, 0),
        buffered=True,
        pixel_perfect_hit=True,
    )
    canvas.add(l_shape)

    # This group automatically calculates its bounds to fit its children.
    shrink_group = ShrinkWrapGroup(selectable=True)

    # Children are positioned relative to the group's origin initially.
    sg_child1 = ExampleElement(10, 20, 80, 50, background=(1, 0.5, 0.5, 1))
    sg_child2 = ExampleElement(120, 90, 60, 100, background=(0.5, 0.5, 1, 1))
    shrink_group.add(sg_child1)
    shrink_group.add(sg_child2)

    # The group itself can be positioned. Its children will move with it.
    shrink_group.set_pos(500, 400)

    # This call calculates the tight bounding box around the children (in their
    # new world positions) and compensates their local transforms so they
    # appear in the same place.
    shrink_group.update_bounds()
    canvas.add(shrink_group)

    # New editable element to test the edit mode feature
    editable = EditableElement(
        550, 600, 200, 150, background=(0.4, 0.4, 0.2, 1), buffered=True
    )
    canvas.add(editable)


app = CanvasApp()
app.run([])
