import cairo
import numpy as np
from gi.repository import Gtk


class HistogramPreview(Gtk.DrawingArea):
    SIZE = 200
    MARGIN = 5

    def __init__(self):
        super().__init__()
        self.histogram: np.ndarray | None = None
        self.min_threshold: float = 0.0
        self.max_threshold: float = 1.0
        self.set_content_width(self.SIZE)
        self.set_content_height(self.SIZE)
        self.set_draw_func(self._draw_func)

    def update_histogram(
        self,
        histogram: np.ndarray | None,
        min_threshold: float,
        max_threshold: float,
    ):
        self.histogram = histogram
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.queue_draw()

    def _draw_func(self, area, ctx: cairo.Context, width: int, height: int):
        ctx.set_source_rgba(0, 0, 0, 0)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)

        if self.histogram is None:
            ctx.set_source_rgba(0.5, 0.5, 0.5, 1.0)
            ctx.set_font_size(12)
            ctx.move_to(width // 2 - 40, height // 2)
            ctx.show_text("No image")
            return

        draw_width = width - 2 * self.MARGIN
        draw_height = height - 2 * self.MARGIN
        max_count = np.max(self.histogram) if np.max(self.histogram) > 0 else 1

        bar_width = draw_width / len(self.histogram)

        color = self.get_color()
        ctx.set_source_rgba(color.red, color.green, color.blue, color.alpha)
        for i, count in enumerate(self.histogram):
            x = self.MARGIN + i * bar_width
            bar_height = (count / max_count) * draw_height
            ctx.rectangle(
                x, height - self.MARGIN - bar_height, bar_width, bar_height
            )
        ctx.fill()

        min_x = self.MARGIN + self.min_threshold * draw_width
        max_x = self.MARGIN + self.max_threshold * draw_width

        ctx.set_source_rgba(0.2, 0.6, 1.0, 0.7)
        ctx.set_line_width(2)
        ctx.move_to(min_x, self.MARGIN)
        ctx.line_to(min_x, height - self.MARGIN)
        ctx.stroke()

        ctx.set_source_rgba(1.0, 0.4, 0.2, 0.7)
        ctx.move_to(max_x, self.MARGIN)
        ctx.line_to(max_x, height - self.MARGIN)
        ctx.stroke()
