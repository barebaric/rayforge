import math
import logging
from typing import Tuple, Generator
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
        y_axis_down: bool = False,
        fg_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        grid_color: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0),
    ):
        self.grid_size_mm: float = grid_size_mm
        self.width_px: int = width_px
        self.height_px: int = height_px
        self.width_mm: float = width_mm
        self.height_mm: float = height_mm
        self.pan_x_mm: float = pan_x_mm
        self.pan_y_mm: float = pan_y_mm
        self.zoom_level: float = zoom_level
        self.y_axis_down: bool = y_axis_down
        self.fg_color: Tuple[float, float, float, float] = fg_color
        self.grid_color: Tuple[float, float, float, float] = grid_color

    def _get_content_layout(self) -> Tuple[float, float, float, float]:
        """
        A centralized helper to calculate the content area's rectangle in
        unzoomed widget pixels, ensuring it respects the mm aspect ratio.
        This is the single source of truth for layout.

        Returns:
            A tuple of (content_x, content_y, content_width, content_height).
        """
        # 1. Calculate space needed for axes and labels.
        x_axis_space = float(self.get_x_axis_height())
        y_axis_space = float(self.get_y_axis_width())

        # Define paddings based on original logic.
        left_padding = y_axis_space
        right_padding = math.ceil(y_axis_space / 2)
        total_horiz_padding = left_padding + right_padding

        if self.y_axis_down:
            top_padding = x_axis_space
            bottom_padding = 0  # Original logic had no bottom padding.
        else:  # Y-up
            top_padding = math.ceil(x_axis_space / 2)
            bottom_padding = x_axis_space
        total_vert_padding = top_padding + bottom_padding

        # 2. Determine the available drawing area after subtracting padding.
        available_width = float(self.width_px) - total_horiz_padding
        available_height = float(self.height_px) - total_vert_padding

        if available_width <= 0 or available_height <= 0:
            logger.warning(
                "Available drawing area is non-positive; "
                "canvas may be too small."
            )
            return left_padding, top_padding, 0.0, 0.0

        # 3. Calculate the target aspect ratio from mm dimensions.
        if self.width_mm <= 0 or self.height_mm <= 0:
            return left_padding, top_padding, available_width, available_height

        world_aspect_ratio = self.width_mm / self.height_mm

        # 4. Calculate content dimensions that fit and match aspect ratio.
        available_aspect_ratio = available_width / available_height

        if available_aspect_ratio > world_aspect_ratio:
            # Available area is wider than needed. Height is the constraint.
            content_height = available_height
            content_width = content_height * world_aspect_ratio
        else:
            # Available area is taller than needed. Width is the constraint.
            content_width = available_width
            content_height = content_width / world_aspect_ratio

        # 5. Center the content area within the available space.
        x_offset = (available_width - content_width) / 2
        y_offset = (available_height - content_height) / 2

        content_x = left_padding + x_offset
        content_y = top_padding + y_offset

        return content_x, content_y, content_width, content_height

    def get_content_size(self) -> Tuple[float, float]:
        """
        Calculates the content area dimensions in pixels, including zoom.
        """
        _, _, content_width_px, content_height_px = self._get_content_layout()
        return (
            content_width_px * self.zoom_level,
            content_height_px * self.zoom_level,
        )

    def get_pixels_per_mm(self) -> Tuple[float, float]:
        """
        Calculates the pixel resolution, taking into account zoom.
        """
        _, _, content_width_px, content_height_px = self._get_content_layout()

        pixels_per_mm_x = (
            (content_width_px / self.width_mm) * self.zoom_level
            if self.width_mm > 0
            else 0.0
        )
        pixels_per_mm_y = (
            (content_height_px / self.height_mm) * self.zoom_level
            if self.height_mm > 0
            else 0.0
        )
        return pixels_per_mm_x, pixels_per_mm_y

    def get_origin(self) -> Tuple[float, float]:
        """
        Calculates the pixel position of the origin (0,0) in the content area.
        """
        content_x, content_y, _, content_height_px = self._get_content_layout()
        pixels_per_mm_x, pixels_per_mm_y = self.get_pixels_per_mm()

        x_px = content_x - self.pan_x_mm * pixels_per_mm_x

        if self.y_axis_down:
            # Y-down: origin (y_mm=0) is at the top of the content area.
            y_px = content_y - self.pan_y_mm * pixels_per_mm_y
        else:
            # Y-up: origin (y_mm=0) is at the bottom of the content area.
            content_bottom_y = content_y + content_height_px
            y_px = content_bottom_y + self.pan_y_mm * pixels_per_mm_y

        return x_px, y_px

    def _x_axis_intervals(self) -> Generator[Tuple[float, float], None, None]:
        """
        Yields (x_mm, x_px) tuples for visible grid lines.
        """
        content_x, _, content_width_px, _ = self._get_content_layout()
        pixels_per_mm_x = content_width_px / self.width_mm * self.zoom_level
        visible_width_mm = self.width_mm / self.zoom_level

        visible_min_x_mm = max(0, self.pan_x_mm)
        visible_max_x_mm = min(self.width_mm, self.pan_x_mm + visible_width_mm)

        k_start = max(
            0, math.ceil(visible_min_x_mm / self.grid_size_mm)
        )
        k_end = math.floor(visible_max_x_mm / self.grid_size_mm)

        for k in range(k_start, k_end + 1):
            x_mm = k * self.grid_size_mm
            if x_mm > self.width_mm:
                break
            x_px = content_x + (x_mm - self.pan_x_mm) * pixels_per_mm_x
            yield x_mm, x_px

    def _y_axis_intervals(self) -> Generator[Tuple[float, float], None, None]:
        """
        Yields (y_mm, y_px) tuples for visible grid lines.
        """
        _, content_y, _, content_height_px = self._get_content_layout()
        pixels_per_mm_y = content_height_px / self.height_mm * self.zoom_level
        visible_height_mm = self.height_mm / self.zoom_level

        visible_min_y_mm = max(0, self.pan_y_mm)
        visible_max_y_mm = min(
            self.height_mm, self.pan_y_mm + visible_height_mm
        )

        k_start = max(
            0, math.ceil(visible_min_y_mm / self.grid_size_mm)
        )
        k_end = math.floor(visible_max_y_mm / self.grid_size_mm)

        for k in range(k_start, k_end + 1):
            y_mm = k * self.grid_size_mm
            if y_mm > self.height_mm:
                break

            if self.y_axis_down:
                # Direct conversion from mm to pixels, origin at top
                y_px = (
                    content_y
                    + (y_mm - self.pan_y_mm) * pixels_per_mm_y
                )
            else:
                # Inverted conversion, origin at bottom (Y-up)
                y_px = (
                    content_y
                    + content_height_px
                    - (y_mm - self.pan_y_mm) * pixels_per_mm_y
                )
            yield y_mm, y_px

    def draw_grid(self, ctx: cairo.Context):
        """Draws the grid lines onto the Cairo context."""
        ctx.save()
        content_x, content_y, content_width, content_height = (
            self._get_content_layout()
        )
        ctx.set_source_rgba(*self.grid_color)
        ctx.set_hairline(True)

        # Vertical lines
        for x_mm, x_px in self._x_axis_intervals():
            ctx.move_to(x_px, content_y)
            ctx.line_to(x_px, content_y + content_height)
            ctx.stroke()

        # Horizontal lines
        for y_mm, y_px in self._y_axis_intervals():
            ctx.move_to(content_x, y_px)
            ctx.line_to(content_x + content_width, y_px)
            ctx.stroke()

        ctx.restore()

    def draw_axes_and_labels(self, ctx: cairo.Context):
        """Draws the axes and labels onto the Cairo context."""
        ctx.save()
        content_x, content_y, content_width, content_height = (
            self._get_content_layout()
        )

        # Draw fixed axis lines and labels
        ctx.set_source_rgba(*self.fg_color)
        ctx.set_line_width(1)

        # Y-axis line (fixed at left of content area)
        ctx.move_to(content_x, content_y)
        ctx.line_to(content_x, content_y + content_height)
        ctx.stroke()

        # X-axis line (fixed at top or bottom of content area)
        if self.y_axis_down:
            x_axis_y_px = content_y
        else:
            x_axis_y_px = content_y + content_height
        ctx.move_to(content_x, x_axis_y_px)
        ctx.line_to(content_x + content_width, x_axis_y_px)
        ctx.stroke()

        # X-axis labels
        for x_mm, x_px in self._x_axis_intervals():
            if x_mm == 0:
                continue
            label = f"{x_mm:.0f}"
            extents = ctx.text_extents(label)
            if self.y_axis_down:
                # Draw labels above the axis line (in the top margin)
                label_y = x_axis_y_px - 4
            else:
                # Draw labels below the axis line (in the bottom margin)
                label_y = x_axis_y_px + extents.height + 4
            ctx.move_to(x_px - extents.width / 2, label_y)
            ctx.show_text(label)

        # Y-axis labels (left of fixed y-axis)
        for y_mm, y_px in self._y_axis_intervals():
            if y_mm == 0:
                continue
            label = f"{y_mm:.0f}"
            extents = ctx.text_extents(label)
            ctx.move_to(
                content_x - extents.width - 4, y_px + extents.height / 2
            )
            ctx.show_text(label)

        ctx.restore()

    def get_x_axis_height(self) -> int:
        """Calculates the maximum height of the X-axis labels."""
        # The height of numeric labels is generally constant for a given font.
        # We can measure a representative character like "8", which usually has
        # the maximum height among digits.
        temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        ctx = cairo.Context(temp_surface)

        extents = ctx.text_extents("8")
        return math.ceil(extents.height) + 4  # adding some margin

    def get_y_axis_width(self) -> int:
        """Calculates the maximum width of the Y-axis labels."""
        # The maximum width is determined by the label with the most digits,
        # which corresponds to the largest coordinate value.
        temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        ctx = cairo.Context(temp_surface)

        # The widest label on the Y-axis will be for the largest coordinate.
        max_y_label = f"{self.height_mm:.0f}"
        extents = ctx.text_extents(max_y_label)

        return math.ceil(extents.width) + 4  # adding some margin

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

    def set_y_axis_down(self, y_axis_down: bool):
        self.y_axis_down = y_axis_down

    def set_fg_color(self, fg_color: Tuple[float, float, float, float]):
        self.fg_color = fg_color

    def set_grid_color(self, grid_color: Tuple[float, float, float, float]):
        self.grid_color = grid_color
