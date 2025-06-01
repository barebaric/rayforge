import math
import logging
import cairo


logger = logging.getLogger(__name__)


class AxisRenderer:
    """
    Helper class to render the grid, axes, and labels on a Cairo context.
    """

    def __init__(
        self,
        grid_size_mm: float = 10.0,
        width_px: int = 1,
        height_px: int = 1,
        width_mm: float = 100.0,
        height_mm: float = 100.0,
        pan_x_mm: float = 0.0,
        pan_y_mm: float = 0.0,
        zoom_level: float = 1.0,
    ):
        self.grid_size_mm: float = grid_size_mm
        self.width_px: int = width_px
        self.height_px: int = height_px
        self.width_mm: float = width_mm
        self.height_mm: float = height_mm
        self.pan_x_mm: float = pan_x_mm
        self.pan_y_mm: float = pan_y_mm
        self.zoom_level: float = zoom_level

    def get_content_size(self) -> tuple[int, int]:
        """
        Calculates the content area dimensions and margins.

        Returns:
            Tuple of (content_width_px, content_height_px).
        """
        x_axis_height = self.get_x_axis_height()
        y_axis_width = self.get_y_axis_width()
        right_margin = math.ceil(y_axis_width / 2)
        top_margin = math.ceil(x_axis_height / 2)

        content_width_px = self.width_px - y_axis_width - right_margin
        content_height_px = self.height_px - x_axis_height - top_margin

        if content_width_px < 0 or content_height_px < 0:
            logger.warning(
                "Content area dimensions are negative; "
                "canvas may be too small."
            )
            content_width_px = max(0, content_width_px)
            content_height_px = max(0, content_height_px)

        return math.ceil(content_width_px * self.zoom_level), math.ceil(
            content_height_px * self.zoom_level
        )

    def get_pixels_per_mm(self) -> tuple[float, float]:
        """
        Calculates the pixel resolution, taking into account the current
        pan and zoom levels.

        Returns:
            Tuple of (pixels_per_mm_x, pixels_per_mm_y).
        """
        y_axis_width = self.get_y_axis_width()
        x_axis_height = self.get_x_axis_height()
        right_margin = math.ceil(y_axis_width / 2)
        top_margin = math.ceil(x_axis_height / 2)

        content_width_px = self.width_px - y_axis_width - right_margin
        content_height_px = self.height_px - x_axis_height - top_margin

        if content_width_px <= 0 or content_height_px <= 0:
            logger.warning(
                "Content area dimensions are non-positive; "
                "canvas may be too small."
            )
            return (0, 0)

        pixels_per_mm_x = (
            (content_width_px / self.width_mm) * self.zoom_level
            if self.width_mm > 0
            else 0
        )
        pixels_per_mm_y = (
            (content_height_px / self.height_mm) * self.zoom_level
            if self.height_mm > 0
            else 0
        )
        return pixels_per_mm_x, pixels_per_mm_y

    def get_origin(self) -> tuple[int, int]:
        """
        Calculates the pixel position of the origin (0,0) in the content area,
        taking into account the current pan and zoom levels.

        Returns:
            Tuple of (x_px, y_px) representing the pixel position of the
            origin as integers.
        """
        y_axis_width = self.get_y_axis_width()
        x_axis_height = self.get_x_axis_height()
        top_margin = math.ceil(x_axis_height / 2)

        content_height_px = self.height_px - x_axis_height - top_margin

        pixels_per_mm_x, pixels_per_mm_y = self.get_pixels_per_mm()
        x_px = y_axis_width - self.pan_x_mm * pixels_per_mm_x
        y_px = top_margin + content_height_px + self.pan_y_mm * pixels_per_mm_y

        return round(x_px), round(y_px)

    def _x_axis_intervals(self):
        """
        Yields (x_mm, x_px) tuples for grid lines within
        [0, self.width_mm] that are visible.
        """
        y_axis_width = self.get_y_axis_width()
        right_margin = math.ceil(y_axis_width / 2)
        content_width_px = self.width_px - y_axis_width - right_margin
        pixels_per_mm_x = content_width_px / self.width_mm * self.zoom_level
        visible_width_mm = self.width_mm / self.zoom_level

        visible_min_x_mm = max(0, self.pan_x_mm)
        visible_max_x_mm = min(self.width_mm, self.pan_x_mm + visible_width_mm)

        k_start = max(
            0, math.ceil(visible_min_x_mm / self.grid_size_mm)
        )  # Ensure no negative k
        k_end = math.floor(visible_max_x_mm / self.grid_size_mm)

        for k in range(k_start, k_end + 1):
            x_mm = k * self.grid_size_mm
            if x_mm > self.width_mm:
                break
            x_px = y_axis_width + (x_mm - self.pan_x_mm) * pixels_per_mm_x
            yield x_mm, x_px

    def _y_axis_intervals(self):
        """
        Yields (y_mm, y_px) tuples for grid lines within
        [0, self.height_mm] that are visible.
        """
        x_axis_height = self.get_x_axis_height()
        top_margin = math.ceil(x_axis_height / 2)
        content_height_px = self.height_px - x_axis_height - top_margin
        pixels_per_mm_y = content_height_px / self.height_mm * self.zoom_level
        visible_height_mm = self.height_mm / self.zoom_level

        visible_min_y_mm = max(0, self.pan_y_mm)
        visible_max_y_mm = min(
            self.height_mm, self.pan_y_mm + visible_height_mm
        )

        k_start = max(
            0, math.ceil(visible_min_y_mm / self.grid_size_mm)
        )  # Ensure no negative k
        k_end = math.floor(visible_max_y_mm / self.grid_size_mm)

        for k in range(k_start, k_end + 1):
            y_mm = k * self.grid_size_mm
            if y_mm > self.height_mm:
                break
            y_px = (
                content_height_px
                - (y_mm - self.pan_y_mm) * pixels_per_mm_y
                + top_margin
            )
            yield y_mm, y_px

    def draw_grid(self, ctx: cairo.Context):
        """
        Draws the grid lines onto the Cairo context. Assumes
        context is already transformed for the worksurface content area.
        The grid lines are drawn in pixel coordinates relative to the
        transformed content area origin.

        Args:
            ctx: The Cairo context to draw on.
        """
        ctx.save()

        # Calculate content area dimensions
        y_axis_width = self.get_y_axis_width()
        right_margin = math.ceil(y_axis_width / 2)
        top_margin = math.ceil(self.get_x_axis_height() / 2)

        # Draw grid lines
        ctx.set_source_rgb(0.9, 0.9, 0.9)
        ctx.set_hairline(True)

        # Vertical lines
        for x_mm, x_px in self._x_axis_intervals():
            ctx.move_to(x_px, top_margin)
            ctx.line_to(x_px, self.height_px - self.get_x_axis_height())
            ctx.stroke()

        # Horizontal lines
        for y_mm, y_px in self._y_axis_intervals():
            ctx.move_to(y_axis_width, y_px)
            ctx.line_to(self.width_px - right_margin, y_px)
            ctx.stroke()

        ctx.restore()

    def draw_axes_and_labels(self, ctx: cairo.Context):
        """
        Draws the axes and labels onto the Cairo context.
        Assumes context is in screen coordinates.

        Args:
            ctx: The Cairo context to draw on.
            width_px: The width of the full drawing area in pixels.
            height_px: The height of the full drawing area in pixels.
        """
        ctx.save()

        # Calculate fixed positions for axis lines
        x_axis_height = self.get_x_axis_height()
        y_axis_width = self.get_y_axis_width()
        right_margin = math.ceil(y_axis_width / 2)
        top_margin = math.ceil(x_axis_height / 2)
        x_axis_y = self.height_px - x_axis_height  # Bottom of content area
        y_axis_x = y_axis_width  # Left edge of content area

        # Draw fixed axis lines
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(1)

        # X-axis line (fixed at bottom)
        ctx.move_to(y_axis_width, x_axis_y)
        ctx.line_to(self.width_px - right_margin, x_axis_y)
        ctx.stroke()

        # Y-axis line (fixed at left)
        ctx.move_to(y_axis_x, top_margin)
        ctx.line_to(y_axis_x, self.height_px - x_axis_height)
        ctx.stroke()

        # X-axis labels (below fixed x-axis)
        for x_mm, x_px in self._x_axis_intervals():
            if x_mm == 0:
                continue  # Skip origin label or handle separately if needed
            label = f"{x_mm:.0f}"
            extents = ctx.text_extents(label)
            ctx.move_to(
                x_px - extents.width / 2, x_axis_y + extents.height + 4
            )
            ctx.show_text(label)

        # Y-axis labels (left of fixed y-axis)
        for y_mm, y_px in self._y_axis_intervals():
            if y_mm == 0:
                continue  # Skip origin label or handle separately if needed
            label = f"{y_mm:.0f}"
            extents = ctx.text_extents(label)
            ctx.move_to(
                y_axis_x - extents.width - 4, y_px + extents.height / 2
            )
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
        return math.ceil(max_height) + 4  # adding some margin

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
            extents = ctx.text_extents(f"{y_mm:.0f}")
            max_width = max(max_width, extents.width)
        return math.ceil(max_width) + 4  # adding some margin

    def set_width_px(self, width_px: int):
        self.width_px = width_px

    def set_height_px(self, height_px: int):
        self.height_px = height_px

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
