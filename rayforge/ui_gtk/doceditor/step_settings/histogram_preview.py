import cairo
import numpy as np
from gi.repository import Gtk
from blinker import Signal


class HistogramPreview(Gtk.DrawingArea):
    WIDTH = 200
    HEIGHT = 100
    MARGIN = 5

    def __init__(self):
        super().__init__()
        self.histogram: np.ndarray | None = None
        self._black_point: int = 0
        self._white_point: int = 255
        self._auto_black_point: int = 0
        self._auto_white_point: int = 255
        self._auto_mode: bool = True
        self._dragging: str | None = None
        self._hovering: str | None = None

        self.black_point_changed = Signal()
        self.white_point_changed = Signal()
        self.auto_mode_changed = Signal()

        self.set_content_width(self.WIDTH)
        self.set_content_height(self.HEIGHT)
        self.set_draw_func(self._draw_func)

        click = Gtk.GestureClick.new()
        click.connect("pressed", self._on_pressed)
        click.connect("released", self._on_released)
        self.add_controller(click)

        motion = Gtk.EventControllerMotion.new()
        motion.connect("motion", self._on_motion)
        motion.connect("leave", self._on_leave)
        self.add_controller(motion)

    @property
    def black_point(self) -> int:
        return self._black_point

    @black_point.setter
    def black_point(self, value: int):
        if self._black_point != value:
            self._black_point = max(0, min(254, value))
            self.queue_draw()

    @property
    def white_point(self) -> int:
        return self._white_point

    @white_point.setter
    def white_point(self, value: int):
        if self._white_point != value:
            self._white_point = max(1, min(255, value))
            self.queue_draw()

    @property
    def auto_mode(self) -> bool:
        return self._auto_mode

    @auto_mode.setter
    def auto_mode(self, value: bool):
        if self._auto_mode != value:
            self._auto_mode = value
            self.queue_draw()

    def set_auto_points(self, black_point: int, white_point: int):
        self._auto_black_point = max(0, min(253, black_point))
        self._auto_white_point = max(
            self._auto_black_point + 2, min(255, white_point)
        )
        if self._auto_mode:
            self.queue_draw()

    def update_histogram(self, histogram: np.ndarray | None):
        self.histogram = histogram
        self.queue_draw()

    def set_points(self, black_point: int, white_point: int):
        self._black_point = max(0, min(254, black_point))
        self._white_point = max(1, min(255, white_point))
        self.queue_draw()

    def _value_to_x(self, value: int, width: int) -> float:
        draw_width = width - 2 * self.MARGIN
        return self.MARGIN + (value / 255.0) * draw_width

    def _x_to_value(self, x: float, width: int) -> int:
        draw_width = width - 2 * self.MARGIN
        ratio = (x - self.MARGIN) / draw_width
        return int(round(ratio * 255))

    def _get_handle_at(
        self, x: float, y: float, width: int, height: int
    ) -> str | None:
        if self._auto_mode:
            black_x = self._value_to_x(self._auto_black_point, width)
            white_x = self._value_to_x(self._auto_white_point, width)
        else:
            black_x = self._value_to_x(self._black_point, width)
            white_x = self._value_to_x(self._white_point, width)

        threshold = 10

        if abs(x - black_x) < threshold:
            return "black"
        elif abs(x - white_x) < threshold:
            return "white"
        return None

    def _on_pressed(self, gesture, n_press, x, y):
        if self._auto_mode:
            return
        width = self.get_width()
        handle = self._get_handle_at(x, y, width, self.get_height())
        if handle:
            self._dragging = handle
            gesture.set_state(Gtk.EventSequenceState.CLAIMED)

    def _on_released(self, gesture, n_press, x, y):
        self._dragging = None

    def _on_motion(self, controller, x, y):
        if self._auto_mode:
            return

        width = self.get_width()
        height = self.get_height()

        if self._dragging:
            value = self._x_to_value(x, width)
            value = max(0, min(255, value))

            if self._dragging == "black":
                new_black = min(value, self._white_point - 1)
                if new_black != self._black_point:
                    self._black_point = new_black
                    self.queue_draw()
                    self.black_point_changed.send(
                        self, black_point=self._black_point
                    )
            else:
                new_white = max(value, self._black_point + 1)
                if new_white != self._white_point:
                    self._white_point = new_white
                    self.queue_draw()
                    self.white_point_changed.send(
                        self, white_point=self._white_point
                    )
        else:
            handle = self._get_handle_at(x, y, width, height)
            if handle != self._hovering:
                self._hovering = handle
                self.queue_draw()

    def _on_leave(self, controller):
        if self._hovering:
            self._hovering = None
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

        if self._auto_mode:
            black_x = self._value_to_x(self._auto_black_point, width)
            white_x = self._value_to_x(self._auto_white_point, width)
        else:
            black_x = self._value_to_x(self._black_point, width)
            white_x = self._value_to_x(self._white_point, width)

        if self._auto_mode:
            ctx.set_source_rgba(0.2, 0.6, 1.0, 0.5)
            ctx.set_dash([4, 4])
        else:
            ctx.set_source_rgba(0.2, 0.6, 1.0, 0.7)
            ctx.set_dash([])
        if self._hovering == "black" or self._dragging == "black":
            ctx.set_line_width(3)
        else:
            ctx.set_line_width(2)
        ctx.move_to(black_x, self.MARGIN)
        ctx.line_to(black_x, height - self.MARGIN)
        ctx.stroke()

        if self._auto_mode:
            ctx.set_source_rgba(1.0, 0.4, 0.2, 0.5)
            ctx.set_dash([4, 4])
        else:
            ctx.set_source_rgba(1.0, 0.4, 0.2, 0.7)
            ctx.set_dash([])
        if self._hovering == "white" or self._dragging == "white":
            ctx.set_line_width(3)
        else:
            ctx.set_line_width(2)
        ctx.move_to(white_x, self.MARGIN)
        ctx.line_to(white_x, height - self.MARGIN)
        ctx.stroke()

        ctx.set_source_rgba(1.0, 1.0, 1.0, 0.3)
        ctx.rectangle(
            black_x, self.MARGIN, white_x - black_x, height - 2 * self.MARGIN
        )
        ctx.fill()
