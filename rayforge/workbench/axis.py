import math
import logging
from typing import Tuple
import cairo
from ..core.matrix import Matrix


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
    ):
        self.grid_size_mm: float = grid_size_mm
        self.width_mm: float = width_mm
        self.height_mm: float = height_mm
        self.y_axis_down: bool = y_axis_down
        self.fg_color: Tuple[float, float, float, float] = fg_color
        self.grid_color: Tuple[float, float, float, float] = grid_color

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
        ctx.save()

        try:
            inv_view = view_transform.invert()
        except Exception:
            ctx.restore()
            return

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

        ctx.set_source_rgba(*self.grid_color)
        ctx.set_hairline(True)

        k_start_x = math.ceil(visible_min_x / self.grid_size_mm)
        k_end_x = math.floor(visible_max_x / self.grid_size_mm)
        for k in range(k_start_x, k_end_x + 1):
            x_mm = k * self.grid_size_mm
            p1_px = view_transform.transform_point((x_mm, visible_min_y))
            p2_px = view_transform.transform_point((x_mm, visible_max_y))
            ctx.move_to(p1_px[0], p1_px[1])
            ctx.line_to(p2_px[0], p2_px[1])
            ctx.stroke()

        k_start_y = math.ceil(visible_min_y / self.grid_size_mm)
        k_end_y = math.floor(visible_max_y / self.grid_size_mm)
        for k in range(k_start_y, k_end_y + 1):
            y_mm = k * self.grid_size_mm
            p1_px = view_transform.transform_point((visible_min_x, y_mm))
            p2_px = view_transform.transform_point((visible_max_x, y_mm))
            ctx.move_to(p1_px[0], p1_px[1])
            ctx.line_to(p2_px[0], p2_px[1])
            ctx.stroke()

        ctx.set_source_rgba(*self.fg_color)
        ctx.set_line_width(1)

        if self.y_axis_down:
            # Y-down view: Origin is top-left.
            # X-axis is at the top of the world area (y = height_mm)
            x_axis_y = self.height_mm
            # Y-axis starts at the top-left and goes down to the bottom-left.
            y_axis_start_mm = (0, self.height_mm)
            y_axis_end_mm = (0, 0)
        else:
            # Y-up view: Origin is bottom-left.
            # X-axis is at the bottom of the world area (y = 0)
            x_axis_y = 0.0
            # Y-axis starts at the bottom-left and goes up.
            y_axis_start_mm = (0, 0)
            y_axis_end_mm = (0, self.height_mm)

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

        # X-axis labels
        for k in range(1, int(self.width_mm / self.grid_size_mm) + 1):
            x_mm = k * self.grid_size_mm
            if x_mm >= self.width_mm:
                break
            label = f"{x_mm:.0f}"
            extents = ctx.text_extents(label)
            label_pos_px = view_transform.transform_point((x_mm, x_axis_y))
            if self.y_axis_down:
                # Y-down: axis is at the top, labels go above it
                y_offset = -4
            else:
                # Y-up: axis is at the bottom, labels go below it
                y_offset = extents.height + 4
            ctx.move_to(
                label_pos_px[0] - extents.width / 2, label_pos_px[1] + y_offset
            )
            ctx.show_text(label)

        for k in range(1, int(self.height_mm / self.grid_size_mm) + 1):
            y_mm = k * self.grid_size_mm
            if y_mm >= self.height_mm:
                break
            label = f"{y_mm:.0f}"
            extents = ctx.text_extents(label)
            # For y-down, a label of "10" means 10mm down from the top.
            world_y = self.height_mm - y_mm if self.y_axis_down else y_mm
            label_pos_px = view_transform.transform_point((0, world_y))
            ctx.move_to(
                label_pos_px[0] - extents.width - 4,
                label_pos_px[1] + extents.height / 2,
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
