# flake8: noqa: E402
import gi
import logging
import gettext
from pathlib import Path

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk
import cairo
from typing import Optional, Dict, Tuple

base_path = Path(__file__).parent
gettext.install("canvas", base_path / "rayforge" / "locale")
logging.basicConfig(level=logging.DEBUG)


from rayforge.workbench.canvas import Canvas, CanvasElement, ShrinkWrapGroup
from rayforge.core.matrix import Matrix


class ExampleElement(CanvasElement):
    """
    A custom canvas element that draws a triangle to clearly show its
    orientation, useful for debugging transformations.
    """

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
        m_content = self.content_transform.m
        cairo_content_matrix = cairo.Matrix(
            m_content[0, 0],
            m_content[1, 0],
            m_content[0, 1],
            m_content[1, 1],
            m_content[0, 2],
            m_content[1, 2],
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
    elem = ExampleElement(100, 120, 100, 100, background=(0.5, 1, 0.5, 1))
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


app = CanvasApp()
app.run([])
