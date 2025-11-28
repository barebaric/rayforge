import math
import logging
from typing import Tuple
import cairo
from ...core.matrix import Matrix


logger = logging.getLogger(__name__)


class AxisRenderer:
    """
    Helper class to render the grid, axes, and labels on a Cairo context.
    This renderer is stateless regarding pan and zoom; it operates in
    world coordinates (mm) and relies on a view_transform matrix to map
    to widget pixel coordinates.
    """

    def __init__(
        self,
        grid_size_mm: float = 10.0,
        width_mm: float = 100.0,
        height_mm: float = 100.0,
        y_axis_down: bool = False,
        fg_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        grid_color: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0),
        show_grid: bool = True,
        show_axis: bool = True,
    ):
        self.grid_size_mm: float = grid_size_mm
        self.width_mm: float = width_mm
        self.height_mm: float = height_mm
        self.y_axis_down: bool = y_axis_down
        self.fg_color: Tuple[float, float, float, float] = fg_color
        self.grid_color: Tuple[float, float, float, float] = grid_color
        self.show_grid: bool = show_grid
        self.show_axis: bool = show_axis
        # Minimum pixel spacing for grid lines to avoid clutter
        self.min_grid_spacing_px = 50.0

    def get_content_layout(
        self, widget_w: int, widget_h: int
    ) -> Tuple[float, float, float, float]:
        """
        Calculates the content area's rectangle in widget pixels, respecting
        the mm aspect ratio. This is the single source of truth for layout.

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
            bottom_padding = math.ceil(x_axis_space / 2)
        else:  # Y-up
            top_padding = math.ceil(x_axis_space / 2)
            bottom_padding = x_axis_space
        total_vert_padding = top_padding + bottom_padding

        # 2. Determine the available drawing area after subtracting padding.
        available_width = float(widget_w) - total_horiz_padding
        available_height = float(widget_h) - total_vert_padding

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

    def get_base_pixels_per_mm(self, widget_w: int, widget_h: int) -> float:
        """
        Calculates the base pixels/mm for a zoom level of 1.0.
        """
        _, _, content_w, content_h = self.get_content_layout(
            widget_w, widget_h
        )
        if self.width_mm <= 0 or self.height_mm <= 0:
            return 1.0

        base_ppm_x = content_w / self.width_mm
        base_ppm_y = content_h / self.height_mm
        return min(base_ppm_x, base_ppm_y)

    def _get_adaptive_grid_size(self, pixels_per_mm: float) -> float:
        """
        Calculates an appropriate grid spacing in mm based on the current
        zoom level (pixels per mm).
        """
        if pixels_per_mm <= 1e-6:
            return self.grid_size_mm

        # Calculate the grid size in mm that would correspond to our desired
        # minimum pixel spacing.
        target_grid_size_mm = self.min_grid_spacing_px / pixels_per_mm

        # Find the next "nice" number (1, 2, 5, 10, 20, 50, 100...) that is
        # greater than or equal to the target size.
        power_of_10 = 10 ** math.floor(math.log10(target_grid_size_mm))

        # Use more balanced thresholds for selecting the multiplier
        relative_size = target_grid_size_mm / power_of_10
        if relative_size <= 1.5:
            return power_of_10
        elif relative_size <= 3.5:
            return 2 * power_of_10
        elif relative_size <= 7.5:
            return 5 * power_of_10
        else:
            return 10 * power_of_10

    def draw_grid_and_labels(
        self,
        ctx: cairo.Context,
        view_transform: Matrix,
        widget_w: int,
        widget_h: int,
    ):
        """
        Draws the grid, axes, and labels onto the Cairo context using the
        provided world-to-view transform and widget dimensions.
        """
        if not self.show_grid and not self.show_axis:
            return

        ctx.save()

        try:
            inv_view = view_transform.invert()
        except Exception:
            ctx.restore()
            return

        # --- Shared Calculations ---
        # Calculate adaptive grid spacing
        scale_x, scale_y = view_transform.get_scale()
        pixels_per_mm = (abs(scale_x) + abs(scale_y)) / 2.0
        adaptive_grid_size_mm = self._get_adaptive_grid_size(pixels_per_mm)

        # Calculate visible bounds in mm for culling/optimizing grid lines
        tl_mm = inv_view.transform_point((0, 0))
        br_mm = inv_view.transform_point((widget_w, widget_h))
        visible_min_x, visible_max_x = (
            min(tl_mm[0], br_mm[0]),
            max(tl_mm[0], br_mm[0]),
        )
        visible_min_y, visible_max_y = (
            min(tl_mm[1], br_mm[1]),
            max(tl_mm[1], br_mm[1]),
        )

        # --- Draw Grid ---
        if self.show_grid:
            self._draw_grid(
                ctx,
                view_transform,
                adaptive_grid_size_mm,
                visible_min_x,
                visible_max_x,
                visible_min_y,
                visible_max_y,
            )

        # --- Draw Axes and Labels ---
        if self.show_axis:
            self._draw_axis_and_labels(
                ctx, view_transform, adaptive_grid_size_mm
            )

        ctx.restore()

    def _draw_grid(
        self,
        ctx: cairo.Context,
        view_transform: Matrix,
        grid_size_mm: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ):
        """Internal helper to draw the infinite grid lines."""
        ctx.set_source_rgba(*self.grid_color)
        ctx.set_hairline(True)

        # Vertical lines (along X)
        k_start_x = math.ceil(min_x / grid_size_mm)
        k_end_x = math.floor(max_x / grid_size_mm)
        for k in range(k_start_x, k_end_x + 1):
            x_mm = k * grid_size_mm
            p1_px = view_transform.transform_point((x_mm, min_y))
            p2_px = view_transform.transform_point((x_mm, max_y))
            ctx.move_to(p1_px[0], p1_px[1])
            ctx.line_to(p2_px[0], p2_px[1])
            ctx.stroke()

        # Horizontal lines (along Y)
        k_start_y = math.ceil(min_y / grid_size_mm)
        k_end_y = math.floor(max_y / grid_size_mm)
        for k in range(k_start_y, k_end_y + 1):
            y_mm = k * grid_size_mm
            p1_px = view_transform.transform_point((min_x, y_mm))
            p2_px = view_transform.transform_point((max_x, y_mm))
            ctx.move_to(p1_px[0], p1_px[1])
            ctx.line_to(p2_px[0], p2_px[1])
            ctx.stroke()

    def _draw_axis_and_labels(
        self,
        ctx: cairo.Context,
        view_transform: Matrix,
        grid_size_mm: float,
    ):
        """Internal helper to draw the main XY axes and text labels."""
        # Calculate precision needed to display fractional grid sizes
        if grid_size_mm < 1:
            precision = int(math.ceil(-math.log10(grid_size_mm)))
        else:
            precision = 0

        ctx.set_source_rgba(*self.fg_color)
        ctx.set_line_width(1)

        # Determine axis positions based on Y orientation
        if self.y_axis_down:
            # Y-down view: Origin top-left.
            # X-axis at top (y = height), Y-axis goes down.
            x_axis_y = self.height_mm
            y_axis_start_mm = (0, self.height_mm)
            y_axis_end_mm = (0, 0)
        else:
            # Y-up view: Origin bottom-left.
            # X-axis at bottom (y = 0), Y-axis goes up.
            x_axis_y = 0.0
            y_axis_start_mm = (0, 0)
            y_axis_end_mm = (0, self.height_mm)

        # Draw Axis Lines
        x_axis_start_mm = (0, x_axis_y)
        x_axis_end_mm = (self.width_mm, x_axis_y)

        x_start_px = view_transform.transform_point(x_axis_start_mm)
        x_end_px = view_transform.transform_point(x_axis_end_mm)
        y_start_px = view_transform.transform_point(y_axis_start_mm)
        y_end_px = view_transform.transform_point(y_axis_end_mm)

        ctx.move_to(x_start_px[0], x_start_px[1])
        ctx.line_to(x_end_px[0], x_end_px[1])
        ctx.stroke()
        ctx.move_to(y_start_px[0], y_start_px[1])
        ctx.line_to(y_end_px[0], y_end_px[1])
        ctx.stroke()

        # Draw X Labels
        k = 1
        while True:
            x_mm = k * grid_size_mm
            if x_mm >= self.width_mm - 1e-6:
                break

            label_val = round(x_mm, precision)
            label = f"{label_val:g}"
            extents = ctx.text_extents(label)
            label_pos_px = view_transform.transform_point((x_mm, x_axis_y))

            if self.y_axis_down:
                y_offset = -4
            else:
                y_offset = extents.height + 4

            ctx.move_to(
                label_pos_px[0] - extents.width / 2,
                label_pos_px[1] + y_offset,
            )
            ctx.show_text(label)
            k += 1

        # Draw Y Labels
        k = 1
        while True:
            y_mm = k * grid_size_mm
            if y_mm >= self.height_mm - 1e-6:
                break

            label_val = round(y_mm, precision)
            label = f"{label_val:g}"

            extents = ctx.text_extents(label)
            # For y-down, a label of "10" means 10mm down from the top.
            world_y = self.height_mm - y_mm if self.y_axis_down else y_mm
            label_pos_px = view_transform.transform_point((0, world_y))

            ctx.move_to(
                label_pos_px[0] - extents.width - 4,
                label_pos_px[1] + extents.height / 2,
            )
            ctx.show_text(label)
            k += 1

    def get_x_axis_height(self) -> int:
        """Calculates the maximum height of the X-axis labels."""
        # The height of numeric labels is generally constant for a given font.
        # We can measure a representative character like "8", which usually has
        # the maximum height among digits.
        temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        ctx = cairo.Context(temp_surface)

        extents = ctx.text_extents("8")
        return math.ceil(extents.height) + 4

    def get_y_axis_width(self) -> int:
        """Calculates the maximum width of the Y-axis labels."""
        # The maximum width is determined by the label with the most digits,
        # which corresponds to the largest coordinate value.
        temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        ctx = cairo.Context(temp_surface)

        # The widest label on the Y-axis will be for the largest coordinate.
        max_y_label = f"{self.height_mm:.0f}"
        extents = ctx.text_extents(max_y_label)
        return math.ceil(extents.width) + 4

    def set_width_mm(self, width_mm: float):
        self.width_mm = width_mm

    def set_height_mm(self, height_mm: float):
        self.height_mm = height_mm

    def set_y_axis_down(self, y_axis_down: bool):
        self.y_axis_down = y_axis_down

    def set_fg_color(self, fg_color: Tuple[float, float, float, float]):
        self.fg_color = fg_color

    def set_grid_color(self, grid_color: Tuple[float, float, float, float]):
        self.grid_color = grid_color
