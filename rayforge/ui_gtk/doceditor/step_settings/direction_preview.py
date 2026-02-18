import math
from gi.repository import Gtk
import cairo


class DirectionPreview(Gtk.DrawingArea):
    VISUAL_SIZE = 160
    MARGIN = 8
    LINE_SPACING = 16
    ARROW_SIZE = 6

    TRAVEL_COLOR = (1.0, 0.4, 0.0, 0.7)
    CUT_COLOR = (1.0, 0.0, 1.0, 1.0)

    def __init__(
        self, direction_degrees: float = 0, cross_hatch: bool = False
    ):
        super().__init__()
        self.direction_degrees = direction_degrees
        self.cross_hatch = cross_hatch
        self.set_content_width(self.VISUAL_SIZE)
        self.set_content_height(self.VISUAL_SIZE)
        self.set_draw_func(self._draw_func)

    def update(self, degrees: float, cross_hatch: bool = False):
        self.direction_degrees = degrees
        self.cross_hatch = cross_hatch
        self.queue_draw()

    def _draw_pass(self, ctx, cx, cy, direction_degrees, num_lines=7):
        angle_rad = math.radians(direction_degrees)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        perp_cos, perp_sin = -sin_a, cos_a

        half_extent = (self.VISUAL_SIZE - 2 * self.MARGIN) / 2 - 10

        line_endpoints = []
        cut_starts = []
        cut_ends = []
        for i in range(num_lines):
            offset = (i - num_lines // 2) * self.LINE_SPACING
            lx = cx + offset * perp_cos
            ly = cy + offset * perp_sin

            start_x = lx - half_extent * cos_a + 10 * cos_a
            start_y = ly - half_extent * sin_a + 10 * sin_a
            end_x = lx + half_extent * cos_a - 10 * cos_a
            end_y = ly + half_extent * sin_a - 10 * sin_a
            line_endpoints.append((start_x, start_y, end_x, end_y))

            direction = 1 if i % 2 == 0 else -1
            if direction > 0:
                cut_starts.append((start_x, start_y))
                cut_ends.append((end_x, end_y))
            else:
                cut_starts.append((end_x, end_y))
                cut_ends.append((start_x, start_y))

        ctx.set_source_rgba(*self.TRAVEL_COLOR)
        ctx.set_line_width(1.5)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for i in range(len(cut_ends) - 1):
            curr_x, curr_y = cut_ends[i]
            next_x, next_y = cut_starts[i + 1]
            ctx.move_to(curr_x, curr_y)
            ctx.line_to(next_x, next_y)
        ctx.stroke()

        ctx.set_source_rgba(*self.CUT_COLOR)
        ctx.set_line_width(2.0)
        for i, (start_x, start_y, end_x, end_y) in enumerate(line_endpoints):
            direction = 1 if i % 2 == 0 else -1
            ctx.move_to(start_x, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()

            arrow_base_x = end_x if direction > 0 else start_x
            arrow_base_y = end_y if direction > 0 else start_y

            ctx.move_to(arrow_base_x, arrow_base_y)
            ctx.line_to(
                arrow_base_x
                + direction * self.ARROW_SIZE * (-cos_a - perp_cos * 0.5),
                arrow_base_y
                + direction * self.ARROW_SIZE * (-sin_a - perp_sin * 0.5),
            )
            ctx.move_to(arrow_base_x, arrow_base_y)
            ctx.line_to(
                arrow_base_x
                + direction * self.ARROW_SIZE * (-cos_a + perp_cos * 0.5),
                arrow_base_y
                + direction * self.ARROW_SIZE * (-sin_a + perp_sin * 0.5),
            )
            ctx.stroke()

    def _draw_func(self, area, ctx, width, height):
        ctx.set_source_rgba(0, 0, 0, 0)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)

        cx, cy = width / 2, height / 2

        if self.cross_hatch:
            self._draw_pass(ctx, cx, cy, self.direction_degrees + 90)

        self._draw_pass(ctx, cx, cy, self.direction_degrees)
