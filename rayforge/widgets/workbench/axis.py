import math
import logging
import cairo
from gi.repository import Pango, PangoCairo  # type: ignore


logger = logging.getLogger(__name__)


class AxisRenderer:
    """
    Helper class to render the grid, axes, and labels on a Cairo context.
    """
    def __init__(
        self,
        grid_size_mm: float = 10.0,
        font_size: int = 10,
        width_mm: float = 100.0,
        height_mm: float = 100.0,
        pan_x_mm: float = 0.0,
        pan_y_mm: float = 0.0,
        zoom_level: float = 1.0,
    ):
        self.grid_size_mm: float = grid_size_mm
        self.axis_thickness_px: int = 25
        self.width_mm: float = width_mm
        self.height_mm: float = height_mm
        self.pan_x_mm: float = pan_x_mm
        self.pan_y_mm: float = pan_y_mm
        self.font_size: int = font_size
        self.zoom_level: float = zoom_level

    def get_grid_bounds(self, width_px: int, height_px: int) -> tuple[int, int, int, int]:
        """Calculates the origin coordinates in pixels."""
        x_axis_height = self.get_x_axis_height()
        y_axis_width = self.get_y_axis_width()
        right_margin = math.ceil(y_axis_width / 2)
        top_margin = math.ceil(x_axis_height / 2)

        content_width_px = width_px - y_axis_width - right_margin
        content_height_px = height_px - x_axis_height - top_margin

        pixels_per_mm_x = content_width_px / self.width_mm * self.zoom_level
        pixels_per_mm_y = content_height_px / self.height_mm * self.zoom_level

        origin_x_px = y_axis_width + round(self.pan_x_mm * pixels_per_mm_x)
        origin_y_px = (
            content_height_px - round(self.pan_y_mm * pixels_per_mm_y) - x_axis_height
        )  # Y inverted

        return origin_x_px, origin_y_px, width_px-right_margin, top_margin

    def _x_axis_intervals(self, width_px: int, height_px: int):
        """Yields (x_mm, x_px) tuples."""
        origin_x, _, max_x, _ = self.get_grid_bounds(width_px, height_px)
        content_width_px = max_x - origin_x
        pixels_per_mm_x = content_width_px / self.width_mm * self.zoom_level

        for x_mm in range(
            0,
            int(self.width_mm) + int(self.grid_size_mm),
            int(self.grid_size_mm),
        ):
            x_px = origin_x + x_mm * pixels_per_mm_x - self.pan_x_mm * pixels_per_mm_x
            #logger.debug(f"pixels_per_mm_x: {pixels_per_mm_x}, x_mm: {x_mm}, x_px: {x_px}")
            yield x_mm, x_px

    def _y_axis_intervals(self, width_px: int, height_px: int):
        """Yields (y_mm, y_px) tuples."""
        _, origin_y, _, max_y = self.get_grid_bounds(width_px, height_px)
        content_height_px = origin_y - max_y
        pixels_per_mm_y = content_height_px / self.height_mm * self.zoom_level

        for y_mm in range(
            0,
            int(self.height_mm) + int(self.grid_size_mm),
            int(self.grid_size_mm),
        ):
            y_px = (
                origin_y
                - (y_mm + self.pan_y_mm) * pixels_per_mm_y
            )
            #logger.debug(f"pixels_per_mm_y: {pixels_per_mm_y}, y_mm: {y_mm}, y_px: {y_px}")
            yield y_mm, y_px

    def draw_grid(
        self,
        ctx: cairo.Context,
        width_px: int,
        height_px: int,
    ):
        """
        Draws the grid lines onto the Cairo context.
        Assumes context is already transformed for the worksurface content area.
        The grid lines are drawn in pixel coordinates relative to the
        transformed content area origin.

        Args:
            ctx: The Cairo context to draw on.
            pixels_per_mm_x: Pixels per mm in x direction for the current zoom.
            pixels_per_mm_y: Pixels per mm in y direction for the current zoom.
            content_width_px: The width of the content area in pixels.
            content_height_px: The height of the content area in pixels.
        """
        ctx.save()

        # Draw grid lines
        ctx.set_source_rgb(0.9, 0.9, 0.9)
        ctx.set_hairline(True)

        origin_x_px, origin_y_px, max_x_px, max_y_px = self.get_grid_bounds(width_px, height_px)
        # Vertical lines
        for x_mm, x_px in self._x_axis_intervals(width_px, height_px):
            ctx.move_to(x_px, origin_y_px)
            ctx.line_to(x_px, max_y_px)
            #logger.debug(f"Vertical: x_px: {x_px}, origin_x_px: {origin_x_px}, origin_y_px: {origin_y_px}, height_px: {height_px}")
            ctx.stroke()

        # Horizontal lines
        for y_mm, y_px in self._y_axis_intervals(width_px, height_px):
            #logger.debug(f"Horizontal: y_px: {y_px}, origin_x_px: {origin_x_px}, origin_y_px: {origin_y_px}, width_px: {width_px}")
            ctx.move_to(origin_x_px, y_px)
            ctx.line_to(max_x_px, y_px)
            ctx.stroke()

        ctx.restore()

    def draw_axes_and_labels(
        self, ctx: cairo.Context, width_px: int, height_px: int
    ):
        """
        Draws the axes and labels onto the Cairo context.
        Assumes context is in screen coordinates.

        Args:
            ctx: The Cairo context to draw on.
            width_px: The width of the full drawing area in pixels.
            height_px: The height of the full drawing area in pixels.
        """
        ctx.save()

        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(1)  # Axes lines are always 1px thick

        # Draw X axis line (at origin_y_px in screen coords) within the content area
        origin_x_px, origin_y_px, max_x_px, max_y_px = self.get_grid_bounds(width_px, height_px)
        ctx.move_to(origin_x_px, origin_y_px)
        ctx.line_to(max_x_px, origin_y_px)
        ctx.stroke()

        # Draw Y axis line (at origin_x_px in screen coords) within the content area
        ctx.move_to(origin_x_px, origin_y_px)
        ctx.line_to(origin_x_px, max_y_px)
        ctx.stroke()

        # Draw labels
        layout = PangoCairo.create_layout(ctx)
        font_desc = Pango.FontDescription.from_string(f"Sans {self.font_size}")
        layout.set_font_description(font_desc)
        ctx.set_source_rgb(0, 0, 0)

        # Draw origin label
        label = "0"
        extents = ctx.text_extents(label)
        ctx.move_to(origin_x_px - extents.width, origin_y_px + extents.height)
        ctx.show_text(label)

        # X axis labels
        for x_mm, x_px in self._x_axis_intervals(width_px, height_px):
            # Skip drawing the label if it's the origin (0, 0)
            if x_mm == 0:
                continue

            # Draw the label
            label = f"{x_mm}"
            extents = ctx.text_extents(label)
            ctx.move_to(x_px - extents.width / 2, origin_y_px + extents.height + 4)
            ctx.show_text(label)

        # Y axis labels
        for y_mm, y_px in self._y_axis_intervals(width_px, height_px):
            # Skip drawing the label if it's the origin (0, 0)
            if y_mm == 0:
                continue

            # Draw the label
            label = f"{y_mm}"
            extents = ctx.text_extents(label)
            ctx.move_to(origin_x_px - extents.width - 2, y_px + extents.height/2)
            ctx.show_text(label)

        ctx.restore()

    def get_x_axis_height(self) -> int:
        """Calculates the maximum height of the X-axis labels."""
        max_height = 0
        temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        ctx = cairo.Context(temp_surface)

        for x_mm in range(
            0,
            int(self.width_mm) + int(self.grid_size_mm),
            int(self.grid_size_mm),
        ):
            if x_mm == 0:
                continue
            extents = ctx.text_extents(f"{x_mm}")
            max_height = max(max_height, extents.height)
        return math.ceil(max_height) + 2  # adding some margin

    def get_y_axis_width(self) -> int:
        """Calculates the maximum width of the Y-axis labels."""
        max_width = 0
        temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        ctx = cairo.Context(temp_surface)

        for y_mm in range(
            0,
            int(self.height_mm) + int(self.grid_size_mm),
            int(self.grid_size_mm),
        ):
            extents = ctx.text_extents(f"{y_mm}")
            max_width = max(max_width, extents.width)
        return math.ceil(max_width) + 2  # adding some margin

    def set_width_mm(self, width_mm: float):
        self.width_mm = width_mm

    def set_height_mm(self, height_mm: float):
        self.height_mm = height_mm

    def set_pan_x_mm(self, pan_x_mm: float):
        self.pan_x_mm = pan_x_mm

    def set_pan_y_mm(self, pan_y_mm: float):
        self.pan_y_mm = pan_y_mm

    def set_zoom(self, zoom_level: float):
        self.zoom_level = zoom_level
