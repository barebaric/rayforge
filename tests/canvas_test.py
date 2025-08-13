# flake8: noqa: E402
import gi
import logging
import gettext
from pathlib import Path

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # type: ignore
import cairo
from typing import Optional

base_path = Path(__file__).parent
gettext.install("canvas", base_path / "rayforge" / "locale")
logging.basicConfig(level=logging.DEBUG)


from rayforge.workbench.canvas import Canvas, CanvasElement
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
        if not self.buffered:
            ctx.set_source_rgba(*self.background)
            ctx.rectangle(0, 0, self.width, self.height)
            ctx.fill()
            self._draw_content(ctx, self.width, self.height)
        else:
            super().draw(ctx)

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
    A custom element that draws a non-symmetric "L" shape. This is the
    definitive test for pixel-perfect hit detection, especially when
    transformations like Y-flipping are involved. Its alpha mask is
    asymmetric both horizontally and vertically.

    It must be used with `buffered=True` and `pixel_perfect_hit=True`.
    """

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        # Create a surface that is fully transparent by default.
        if width <= 0 or height <= 0:
            return None
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        # --- Define L-Shape Geometry ---
        # Use padding to make it look centered in its bounding box
        padding = min(width, height) * 0.1
        thickness = min(width, height) * 0.3

        # Set the color for the shape
        ctx.set_source_rgba(0.8, 0.2, 0.8, 1.0)  # A vibrant purple

        # Draw the path of the L shape
        ctx.move_to(padding, padding)  # Top-left of vertical bar
        ctx.line_to(padding, height - padding)  # Bottom-left corner
        ctx.line_to(width - padding, height - padding)  # Bottom-right corner
        ctx.line_to(
            width - padding, height - padding - thickness
        )  # Top of horizontal bar
        ctx.line_to(
            padding + thickness, height - padding - thickness
        )  # Inner corner
        ctx.line_to(
            padding + thickness, padding
        )  # Top-right of vertical bar
        ctx.close_path()

        # Fill the defined path
        ctx.fill()

        return surface


class CanvasApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.CanvasApp")

    def do_activate(self):
        win = Gtk.ApplicationWindow(application=self)
        win.set_default_size(1650, 850)
        win.set_title("Canvas Test (Normal Y-Down | Flipped Y-Up)")

        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        win.set_child(main_box)

        canvas_size = 800

        # --- Canvas 1: Normal Coordinates (Y-axis points down) ---
        canvas_normal = Canvas()
        canvas_normal.set_size_request(canvas_size, canvas_size)
        main_box.append(canvas_normal)

        # --- Canvas 2: Flipped Coordinates (Y-axis points up) ---
        canvas_flipped = Canvas()
        canvas_flipped.set_size_request(canvas_size, canvas_size)

        m_scale = Matrix.scale(1, -1)
        m_trans = Matrix.translation(0, canvas_size)
        canvas_flipped.view_transform = m_trans @ m_scale
        main_box.append(canvas_flipped)

        def populate_canvas(canvas: Canvas):
            """Helper function to add the same elements to a canvas."""
            elem = ExampleElement(
                100, 120, 100, 100, background=(0.5, 1, 0.5, 1)
            )
            canvas.add(elem)

            group = CanvasElement(
                250, 250, 400, 350, background=(0, 1, 1, 0.2)
            )
            group.add(
                ExampleElement(
                    20, 20, 100, 80, background=(0.7, 0.7, 1, 1), selectable=False
                )
            )
            group.add(
                ExampleElement(
                    150, 40, 150, 150, background=(0.7, 1, 0.7, 1), buffered=True
                )
            )
            group.add(
                ExampleElement(50, 180, 120, 120, background=(1, 0.7, 1, 1))
            )
            canvas.add(group)

            # The definitive test element for pixel-perfect hit detection.
            # Its silhouette is non-symmetric, proving that hit-testing
            # correctly handles Y-flipped coordinates. You should not be
            # able to select it by clicking in the empty top-right area
            # of its bounding box.
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

        populate_canvas(canvas_normal)
        populate_canvas(canvas_flipped)

        win.present()


app = CanvasApp()
app.run([])
