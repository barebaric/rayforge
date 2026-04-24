import math
import logging
from typing import Optional, Tuple, Union

import cairo
from gi.repository import Gdk, Graphene, Gtk
from blinker import Signal

from rayforge.core.group import Group
from rayforge.core.matrix import Matrix
from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.canvas import Canvas

logger = logging.getLogger(__name__)

MARKER_RADIUS = 4.0
HIT_RADIUS = 12.0
POINT_COLOR_1 = (0.18, 0.80, 0.44, 0.9)
POINT_COLOR_2 = (0.29, 0.56, 0.85, 0.9)
DASH_COLOR = (0.5, 0.5, 0.5, 0.7)

MIN_ZOOM_FACTOR = 0.1
MAX_PIXELS_PER_MM = 100.0


class PickSurface(Canvas):
    """A canvas for picking alignment points on a workpiece or group image.

    Displays the item at its natural size fitted to view and reports
    click coordinates in normalized item-local space (0-1, origin at
    bottom-left, Y-up). Supports scroll-to-zoom and middle-button panning.
    """

    def __init__(self, item: Union[WorkPiece, Group], **kwargs):
        super().__init__(**kwargs)
        self._item = item
        self._is_group = isinstance(item, Group)
        nw, nh = item.natural_size
        self._width_mm = max(nw, 1e-9)
        self._height_mm = max(nh, 1e-9)

        self.zoom_level: float = 1.0
        self.pan_x_mm: float = 0.0
        self.pan_y_mm: float = 0.0
        self._base_scale: float = 0.0
        self._base_offset_x: float = 0.0
        self._base_offset_y: float = 0.0
        self._base_img_h: float = 0.0

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

        self._pick_drag_gesture = Gtk.GestureDrag.new()
        self._pick_drag_gesture.set_button(Gdk.BUTTON_PRIMARY)
        self._pick_drag_gesture.connect("drag-begin", self._on_drag_begin)
        self._pick_drag_gesture.connect("drag-update", self._on_drag_update)
        self._pick_drag_gesture.connect("drag-end", self._on_drag_end)
        self.add_controller(self._pick_drag_gesture)

        self._pan_gesture = Gtk.GestureDrag.new()
        self._pan_gesture.set_button(Gdk.BUTTON_MIDDLE)
        self._pan_gesture.connect("drag-begin", self._on_pan_begin)
        self._pan_gesture.connect("drag-update", self._on_pan_update)
        self.add_controller(self._pan_gesture)

        self._scroll_controller = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
        )
        self._scroll_controller.connect("scroll", self._on_scroll)
        self.add_controller(self._scroll_controller)

        self._motion_controller = Gtk.EventControllerMotion()
        self._motion_controller.connect("motion", self._on_motion)
        self.add_controller(self._motion_controller)

        self._hover_pos: Optional[Tuple[float, float]] = None
        self._drag_start_px: Optional[Tuple[float, float]] = None
        self._cached_surface: Optional[cairo.ImageSurface] = None
        self._cached_ppmm: float = 0.0
        self._pan_start_x_mm: float = 0.0
        self._pan_start_y_mm: float = 0.0

        self.set_cursor(Gdk.Cursor.new_from_name("crosshair"))

    def _rebuild_view_transform(self):
        widget_w, widget_h = self.get_width(), self.get_height()
        if widget_w == 0 or widget_h == 0:
            return

        scale_x = widget_w / self._width_mm
        scale_y = widget_h / self._height_mm
        base_scale = min(scale_x, scale_y)

        img_w = self._width_mm * base_scale
        img_h = self._height_mm * base_scale
        base_offset_x = (widget_w - img_w) / 2
        base_offset_y = (widget_h - img_h) / 2

        self._base_scale = base_scale
        self._base_offset_x = base_offset_x
        self._base_offset_y = base_offset_y
        self._base_img_h = img_h

        m_pan = Matrix.translation(-self.pan_x_mm, -self.pan_y_mm)
        m_scale = Matrix.translation(0, img_h) @ Matrix.scale(
            base_scale, -base_scale
        )
        m_zoom = Matrix.scale(self.zoom_level, self.zoom_level)
        m_offset = Matrix.translation(base_offset_x, base_offset_y)

        self.view_transform = m_offset @ m_zoom @ m_scale @ m_pan

        new_ppmm = base_scale * self.zoom_level
        if abs(new_ppmm - self._cached_ppmm) > 0.01:
            self._cached_surface = None
            self._cached_ppmm = new_ppmm

        self.queue_draw()

    def _get_image_surface(self) -> Optional[cairo.ImageSurface]:
        if self._cached_surface is not None:
            return self._cached_surface
        ppmm = self._cached_ppmm
        if ppmm <= 0:
            return None
        w = max(int(self._width_mm * ppmm), 1)
        h = max(int(self._height_mm * ppmm), 1)
        self._cached_surface = self._item.render_to_pixels(w, h)
        return self._cached_surface

    def do_size_allocate(self, width, height, baseline):
        super().do_size_allocate(width, height, baseline)
        self._rebuild_view_transform()

    def do_snapshot(self, snapshot: Gtk.Snapshot) -> None:
        width = self.get_width()
        height = self.get_height()
        ctx = snapshot.append_cairo(Graphene.Rect().init(0, 0, width, height))

        img_surface = self._get_image_surface()
        if img_surface is not None:
            ctx.save()
            cairo_matrix = cairo.Matrix(*self.view_transform.for_cairo())
            ctx.transform(cairo_matrix)
            ctx.translate(0, self._height_mm)
            ctx.scale(1, -1)
            img_w = img_surface.get_width()
            img_h = img_surface.get_height()
            if img_w > 0 and img_h > 0:
                ctx.scale(
                    self._width_mm / img_w,
                    self._height_mm / img_h,
                )
            ctx.set_source_surface(img_surface, 0, 0)
            ctx.paint()
            ctx.restore()

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
            mm_x, mm_y = self._get_world_coords(hx, hy)
            norm_x = mm_x / self._width_mm
            norm_y = mm_y / self._height_mm
            self._draw_crosshair(ctx, norm_x, norm_y)

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

    def _on_scroll(self, controller, dx, dy):
        zoom_speed = 0.1
        desired_zoom = self.zoom_level * (
            (1 - zoom_speed) if dy > 0 else (1 + zoom_speed)
        )

        if self._base_scale <= 0:
            return
        base_ppm = self._base_scale
        min_ppm = base_ppm * MIN_ZOOM_FACTOR
        max_ppm = MAX_PIXELS_PER_MM
        clamped_ppm = max(min_ppm, min(base_ppm * desired_zoom, max_ppm))
        final_zoom = clamped_ppm / base_ppm
        if abs(final_zoom - self.zoom_level) < 1e-9:
            return

        if self._hover_pos is not None:
            mx, my = self._hover_pos
            focus_x, focus_y = self._get_world_coords(mx, my)
            self.zoom_level = final_zoom
            self._rebuild_view_transform()
            new_x, new_y = self._get_world_coords(mx, my)
            self.pan_x_mm += focus_x - new_x
            self.pan_y_mm += focus_y - new_y
        else:
            self.zoom_level = final_zoom

        self._rebuild_view_transform()

    def _on_pan_begin(self, gesture, start_x, start_y):
        self._pan_start_x_mm = self.pan_x_mm
        self._pan_start_y_mm = self.pan_y_mm

    def _on_pan_update(self, gesture, offset_x, offset_y):
        if self._base_scale <= 0:
            return
        scale = self._base_scale * self.zoom_level
        self.pan_x_mm = self._pan_start_x_mm - offset_x / scale
        self.pan_y_mm = self._pan_start_y_mm + offset_y / scale
        self._rebuild_view_transform()

    def _hit_test_point(self, px: float, py: float) -> Optional[int]:
        if self._point1 is not None:
            sx, sy = self._local_to_pixel(*self._point1)
            if math.hypot(px - sx, py - sy) <= HIT_RADIUS:
                return 0
        if self._point2 is not None:
            sx, sy = self._local_to_pixel(*self._point2)
            if math.hypot(px - sx, py - sy) <= HIT_RADIUS:
                return 1
        return None

    def _on_click(self, gesture, n_press, x, y):
        if self._dragging is not None:
            return

        hit = self._hit_test_point(x, y)
        if hit is not None:
            return

        mm_x, mm_y = self._get_world_coords(x, y)
        norm_x = mm_x / self._width_mm
        norm_y = mm_y / self._height_mm

        if self._pick_phase == 0:
            self._point1 = (norm_x, norm_y)
            self._pick_phase = 1
            self.point_picked.send(self, index=0, x=norm_x, y=norm_y)
        elif self._pick_phase == 1:
            self._point2 = (norm_x, norm_y)
            self._pick_phase = 2
            self.point_picked.send(self, index=1, x=norm_x, y=norm_y)

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
        mm_x, mm_y = self._get_world_coords(current_x, current_y)
        norm_x = mm_x / self._width_mm
        norm_y = mm_y / self._height_mm

        if self._dragging == 0:
            self._point1 = (norm_x, norm_y)
        else:
            self._point2 = (norm_x, norm_y)

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

    def _local_to_pixel(self, norm_x, norm_y):
        mm_x = norm_x * self._width_mm
        mm_y = norm_y * self._height_mm
        return self.view_transform.transform_point((mm_x, mm_y))

    def _draw_marker(self, ctx, local_pos, color):
        px, py = self._local_to_pixel(*local_pos)
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
        px1, py1 = self._local_to_pixel(*p1)
        px2, py2 = self._local_to_pixel(*p2)

        ctx.save()
        ctx.set_source_rgba(*DASH_COLOR)
        ctx.set_line_width(1.5)
        ctx.set_dash((6, 4))
        ctx.move_to(px1, py1)
        ctx.line_to(px2, py2)
        ctx.stroke()
        ctx.restore()

    def _draw_crosshair(self, ctx, lx, ly):
        px, py = self._local_to_pixel(lx, ly)
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
