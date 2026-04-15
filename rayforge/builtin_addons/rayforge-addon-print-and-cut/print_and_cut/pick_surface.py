import math
import logging
from typing import Optional, Tuple

from gi.repository import Gdk, Graphene, Gtk
from blinker import Signal

from rayforge.ui_gtk.canvas import WorldSurface
from rayforge.ui_gtk.canvas2d.elements.workpiece import WorkPieceElement

logger = logging.getLogger(__name__)

MARKER_RADIUS = 4.0
HIT_RADIUS = 12.0
POINT_COLOR_1 = (0.18, 0.80, 0.44, 0.9)
POINT_COLOR_2 = (0.29, 0.56, 0.85, 0.9)
DASH_COLOR = (0.5, 0.5, 0.5, 0.7)


class PickSurface(WorldSurface):
    """A minimal canvas for picking alignment points on a workpiece."""

    def __init__(self, **kwargs):
        super().__init__(
            show_grid=True,
            show_axis=True,
            **kwargs,
        )
        self.remove_controller(self._drag_gesture)
        self._ops_suppressed: bool = True
        self._workpiece_elem: Optional[WorkPieceElement] = None

        self._pick_phase: int = 0
        self._point1: Optional[Tuple[float, float]] = None
        self._point2: Optional[Tuple[float, float]] = None
        self._dragging: Optional[int] = None

        self.point_picked = Signal()
        self.points_reset = Signal()
        self.points_changed = Signal()

        self._click_gesture = Gtk.GestureClick()
        self._click_gesture.set_button(Gdk.BUTTON_PRIMARY)
        self._click_gesture.connect("pressed", self._on_click)
        self._click_gesture.set_propagation_phase(Gtk.PropagationPhase.BUBBLE)
        self.add_controller(self._click_gesture)

        self._drag_gesture = Gtk.GestureDrag.new()
        self._drag_gesture.set_button(Gdk.BUTTON_PRIMARY)
        self._drag_gesture.connect("drag-begin", self._on_drag_begin)
        self._drag_gesture.connect("drag-update", self._on_drag_update)
        self._drag_gesture.connect("drag-end", self._on_drag_end)
        self.add_controller(self._drag_gesture)

        self._motion_controller = Gtk.EventControllerMotion()
        self._motion_controller.connect("motion", self._on_motion)
        self.add_controller(self._motion_controller)

        self._hover_pos: Optional[Tuple[float, float]] = None
        self._drag_start_px: Optional[Tuple[float, float]] = None

        self.set_cursor(Gdk.Cursor.new_from_name("crosshair"))

    def set_workpiece_elem(self, elem: WorkPieceElement):
        self._workpiece_elem = elem
        elem.selectable = False
        elem.show_selection_frame = False
        self.root.add(elem)

    @property
    def ops_suppressed(self) -> bool:
        return self._ops_suppressed

    def get_global_tab_visibility(self) -> bool:
        return False

    def get_view_scale(self):
        return super().get_view_scale()

    def _rebuild_view_transform(self) -> bool:
        scale_changed = super()._rebuild_view_transform()
        if scale_changed and self._workpiece_elem is not None:
            ppm_x, _ = self.get_view_scale()
            self._workpiece_elem.trigger_view_update(ppm_x)
        return scale_changed

    @property
    def point1(self) -> Optional[Tuple[float, float]]:
        return self._point1

    @property
    def point2(self) -> Optional[Tuple[float, float]]:
        return self._point2

    @property
    def is_complete(self) -> bool:
        return self._pick_phase >= 2

    def reset(self):
        self._pick_phase = 0
        self._point1 = None
        self._point2 = None
        self._dragging = None
        self.points_reset.send(self)
        self.queue_draw()

    def set_points(
        self,
        p1: Optional[Tuple[float, float]],
        p2: Optional[Tuple[float, float]],
    ):
        self._point1 = p1
        self._point2 = p2
        if p1 is not None and p2 is not None:
            self._pick_phase = 2
        elif p1 is not None:
            self._pick_phase = 1
        else:
            self._pick_phase = 0
        self.queue_draw()

    def _hit_test_point(self, px: float, py: float) -> Optional[int]:
        if self._point1 is not None:
            sx, sy = self._world_to_pixel(*self._point1)
            if math.hypot(px - sx, py - sy) <= HIT_RADIUS:
                return 0
        if self._point2 is not None:
            sx, sy = self._world_to_pixel(*self._point2)
            if math.hypot(px - sx, py - sy) <= HIT_RADIUS:
                return 1
        return None

    def _on_click(self, gesture, n_press, x, y):
        if self._dragging is not None:
            return

        hit = self._hit_test_point(x, y)
        if hit is not None:
            return

        world_x, world_y = self._get_world_coords(x, y)

        if self._pick_phase == 0:
            self._point1 = (world_x, world_y)
            self._pick_phase = 1
            self.point_picked.send(self, index=0, x=world_x, y=world_y)
        elif self._pick_phase == 1:
            self._point2 = (world_x, world_y)
            self._pick_phase = 2
            self.point_picked.send(self, index=1, x=world_x, y=world_y)

        self.queue_draw()

    def _on_drag_begin(self, gesture, start_x, start_y):
        hit = self._hit_test_point(start_x, start_y)
        if hit is not None:
            self._dragging = hit
            self._drag_start_px = (start_x, start_y)

    def _on_drag_update(self, gesture, offset_x, offset_y):
        if self._dragging is None or self._drag_start_px is None:
            return

        sx, sy = self._drag_start_px
        current_x = sx + offset_x
        current_y = sy + offset_y
        world_x, world_y = self._get_world_coords(current_x, current_y)

        if self._dragging == 0:
            self._point1 = (world_x, world_y)
        else:
            self._point2 = (world_x, world_y)

        self.points_changed.send(self)
        self.queue_draw()

    def _on_drag_end(self, gesture, offset_x, offset_y):
        if self._dragging is not None:
            self._on_drag_update(gesture, offset_x, offset_y)
        self._dragging = None
        self._drag_start_px = None

    def _on_motion(self, controller, x, y):
        self._hover_pos = (x, y)
        self.queue_draw()

    def do_snapshot(self, snapshot: Gtk.Snapshot) -> None:
        super().do_snapshot(snapshot)

        width = self.get_width()
        height = self.get_height()
        ctx = snapshot.append_cairo(Graphene.Rect().init(0, 0, width, height))

        if self._point1 is not None:
            self._draw_marker(ctx, self._point1, POINT_COLOR_1)
        if self._point2 is not None:
            self._draw_marker(ctx, self._point2, POINT_COLOR_2)
        if self._point1 is not None and self._point2 is not None:
            self._draw_dashed_line(ctx, self._point1, self._point2)
        if (
            self._hover_pos is not None
            and self._pick_phase < 2
            and self._dragging is None
        ):
            hx, hy = self._hover_pos
            wx, wy = self._get_world_coords(hx, hy)
            self._draw_crosshair(ctx, wx, wy)

    def _world_to_pixel(self, wx, wy):
        return self.view_transform.transform_point((wx, wy))

    def _draw_marker(self, ctx, world_pos, color):
        px, py = self._world_to_pixel(*world_pos)
        r = MARKER_RADIUS

        ctx.save()
        ctx.arc(px, py, r, 0, 2 * math.pi)
        ctx.set_source_rgba(*color)
        ctx.fill_preserve()
        ctx.set_source_rgba(1, 1, 1, 1)
        ctx.set_line_width(1.5)
        ctx.stroke()
        ctx.restore()

    def _draw_dashed_line(self, ctx, p1, p2):
        px1, py1 = self._world_to_pixel(*p1)
        px2, py2 = self._world_to_pixel(*p2)

        ctx.save()
        ctx.set_source_rgba(*DASH_COLOR)
        ctx.set_line_width(1.5)
        ctx.set_dash((6, 4))
        ctx.move_to(px1, py1)
        ctx.line_to(px2, py2)
        ctx.stroke()
        ctx.restore()

    def _draw_crosshair(self, ctx, wx, wy):
        px, py = self._world_to_pixel(wx, wy)
        size = 10

        ctx.save()
        ctx.set_source_rgba(0.9, 0.9, 0.9, 0.8)
        ctx.set_line_width(1.0)
        ctx.move_to(px - size, py)
        ctx.line_to(px + size, py)
        ctx.move_to(px, py - size)
        ctx.line_to(px, py + size)
        ctx.stroke()
        ctx.restore()
