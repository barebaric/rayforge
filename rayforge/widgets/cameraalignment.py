import logging
import math
from typing import List, Optional, Tuple
import numpy as np
from gi.repository import Gtk, Adw, Gdk, GLib
from ..models.camera import Camera, Pos
from .cameradisplay import CameraDisplay
from .pointbubblewidget import PointBubbleWidget

logger = logging.getLogger(__name__)


class CameraAlignmentDialog(Adw.Window):
    def __init__(self, parent: Gtk.Window, camera: Camera, **kwargs):
        super().__init__(
            transient_for=parent,
            modal=True,
            default_width=1280,
            default_height=960,
            **kwargs,
        )

        self.camera = camera
        self.image_points: List[Optional[Pos]] = []
        self.world_points: List[Pos] = []
        self.active_point_index = -1
        self.dragging_point_index = -1
        self.drag_start_image_x = 0
        self.drag_start_image_y = 0
        self._display_ready = False

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(content)

        header_bar = Adw.HeaderBar()
        header_bar.set_title_widget(
            Adw.WindowTitle(
                title=f"{camera.name} – Image Alignment",
                subtitle=""
            )
        )
        content.append(header_bar)

        vbox = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_top=12,
            margin_bottom=12,
            margin_start=12,
            margin_end=12,
        )
        content.append(vbox)

        self.overlay = Gtk.Overlay()
        self.camera_display = CameraDisplay(camera)
        self.overlay.set_child(self.camera_display)
        vbox.append(self.overlay)

        # um só bubble
        self.bubble = PointBubbleWidget(0)
        self.overlay.add_overlay(self.bubble)
        self.bubble.set_halign(Gtk.Align.START)
        self.bubble.set_valign(Gtk.Align.START)
        self.bubble.set_visible(False)
        self.bubble.value_changed.connect(self.update_apply_button_sensitivity)
        self.bubble.delete_requested.connect(self.on_point_delete_requested)
        self.bubble.focus_requested.connect(self.on_bubble_focus_requested)

        bottom_bar = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=12, margin_top=12
        )
        vbox.append(bottom_bar)

        info_label = Gtk.Label(
            label="Click the image to add new reference points. "
            "Click or drag existing points to edit them.",
            xalign=0,
        )
        info_label.set_wrap(True)
        info_label.set_hexpand(True)
        bottom_bar.append(info_label)

        btn_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            halign=Gtk.Align.END,
        )
        bottom_bar.append(btn_box)

        for label, cb in [
            ("Reset Points", self.on_reset_points_clicked),
            ("Clear All Points", self.on_clear_all_points_clicked),
            ("Cancel", self.on_cancel_clicked),
        ]:
            btn = Gtk.Button(label=label)
            btn.add_css_class("flat")
            btn.connect("clicked", cb)
            btn_box.append(btn)
        self.apply_button = Gtk.Button(label="Apply")
        self.apply_button.add_css_class("suggested-action")
        self.apply_button.connect("clicked", self.on_apply_clicked)
        btn_box.append(self.apply_button)

        # Attach gestures to the camera_display, not the overlay.
        click = Gtk.GestureClick.new()
        click.set_button(Gdk.BUTTON_PRIMARY)
        click.connect("pressed", self.on_image_click)
        self.camera_display.add_controller(click)

        drag = Gtk.GestureDrag.new()
        drag.set_button(Gdk.BUTTON_PRIMARY)
        drag.connect("drag-begin", self.on_drag_begin)
        drag.connect("drag-update", self.on_drag_update)
        drag.connect("drag-end", self.on_drag_end)
        self.camera_display.add_controller(drag)

        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self.on_key_pressed)
        self.add_controller(key_controller)

        # quando display pronto/resized
        self.camera_display.connect("realize", self._on_display_ready)
        self.camera_display.connect(
            "resize", lambda w, x, y: self._on_display_ready()
        )

        # init pontos
        if camera.image_to_world:
            img_pts, wld_pts = camera.image_to_world
            self.image_points, self.world_points = list(img_pts), list(wld_pts)
        else:
            self.image_points = [None] * 4
            self.world_points = [(0.0, 0.0)] * 4

        self.set_active_point(0)
        self.update_apply_button_sensitivity()

    def _on_display_ready(self, *args):
        if not self._display_ready:
            self._display_ready = True
            GLib.idle_add(self._position_bubble)
        else:
            self._position_bubble()

    def _position_bubble(self) -> bool:
        if not self._display_ready or self.active_point_index < 0:
            return False
        coords = self.image_points[self.active_point_index]
        if coords is None:
            return False
        img_x, img_y = coords
        dw, dh = (
            self.camera_display.get_width(),
            self.camera_display.get_height(),
        )
        if dw <= 0 or dh <= 0:
            return True  # Try again
        sw, sh = self.camera.resolution
        dx = img_x * (dw / sw)
        dy = dh - (img_y * (dh / sh))
        alloc = self.bubble.get_allocation()
        bw, bh = alloc.width, alloc.height
        if bw <= 0 or bh <= 0:
            return True  # Try again
        x = max(0, min(dx - bw / 2, dw - bw))
        y = dy + 10
        if y + bh > dh:
            y = max(0, dy - bh - 10)
        self.bubble.set_margin_start(int(x))
        self.bubble.set_margin_top(int(y))
        if not self.bubble.get_visible():
            self.bubble.set_visible(True)
        return False  # Success, do not repeat

    def set_active_point(self, index: int, widget=None):
        if index < 0 or index >= len(self.image_points):
            self.active_point_index = -1
            self.bubble.set_visible(False)
            self.camera_display.set_marked_points(self.image_points, -1)
            return
        self.active_point_index = index
        self.bubble.point_index = index
        img = self.image_points[index]
        if img:
            self.bubble.set_image_coords(*img)
        self.bubble.set_world_coords(*self.world_points[index])
        GLib.idle_add(self._position_bubble)
        (widget or self.bubble.world_x_spin).grab_focus()
        self.camera_display.set_marked_points(self.image_points, index)

    def on_bubble_focus_requested(self, bubble, widget):
        self.set_active_point(self.active_point_index, widget)

    def on_image_click(self, gesture, n, x, y):
        if gesture.get_current_button() != Gdk.BUTTON_PRIMARY:
            return
        ix, iy = self._display_to_image_coords(x, y)
        idx = self._find_point_near(ix, iy)
        if idx >= 0:
            self.set_active_point(idx)
        else:
            self.image_points.append((ix, iy))
            self.world_points.append((0.0, 0.0))
            self.set_active_point(len(self.image_points) - 1)
        self.camera_display.set_marked_points(
            self.image_points, self.active_point_index
        )
        self.update_apply_button_sensitivity()

    def on_drag_begin(self, gesture, x, y):
        ix, iy = self._display_to_image_coords(x, y)
        idx = self._find_point_near(ix, iy)
        if idx >= 0:
            self.dragging_point_index = idx
            self.drag_start_image_x, self.drag_start_image_y = (
                self.image_points[idx]
            )
            gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        else:
            self.dragging_point_index = -1
            gesture.set_state(Gtk.EventSequenceState.DENIED)

    def on_drag_update(self, gesture, dx, dy):
        idx = self.dragging_point_index
        if idx < 0:
            return
        dw, dh = (
            self.camera_display.get_width(),
            self.camera_display.get_height(),
        )
        iw, ih = self.camera.resolution
        sx, sy = (dw / iw if dw > 0 else 1), (dh / ih if dh > 0 else 1)
        nx = self.drag_start_image_x + dx / sx
        ny = self.drag_start_image_y - dy / sy
        self.image_points[idx] = (nx, ny)
        if idx == self.active_point_index:
            self.bubble.set_image_coords(nx, ny)
            self._position_bubble()
        self.camera_display.set_marked_points(
            self.image_points, self.active_point_index
        )
        self.camera_display.queue_draw()

    def on_drag_end(self, gesture, dx, dy):
        if self.dragging_point_index >= 0:
            self.dragging_point_index = -1

    def on_key_pressed(self, controller, keyval, keycode, state):
        if keyval == Gdk.KEY_Escape:
            self.close()
            return Gdk.EVENT_STOP

        return Gdk.EVENT_PROPAGATE

    def on_reset_points_clicked(self, _):
        self.image_points.clear()
        self.world_points.clear()
        if self.camera.image_to_world:
            img, wld = self.camera.image_to_world
            self.image_points, self.world_points = list(img), list(wld)
        else:
            self.image_points = [None] * 4
            self.world_points = [(0.0, 0.0)] * 4
        self.set_active_point(0)
        self.update_apply_button_sensitivity()

    def on_clear_all_points_clicked(self, _):
        self.image_points.clear()
        self.world_points.clear()
        self.set_active_point(-1)
        self.update_apply_button_sensitivity()

    def on_point_delete_requested(self, bubble):
        i = bubble.point_index
        if 0 <= i < len(self.image_points):
            self.image_points.pop(i)
            self.world_points.pop(i)
        if self.image_points:
            self.set_active_point(min(i, len(self.image_points) - 1))
        else:
            self.set_active_point(-1)
        self.camera_display.set_marked_points(
            self.image_points, self.active_point_index
        )
        self.update_apply_button_sensitivity()

    def update_apply_button_sensitivity(self, *_):
        if self.active_point_index >= 0 and self.active_point_index < len(
            self.world_points
        ):
            self.world_points[self.active_point_index] = (
                self.bubble.get_world_coords()
            )

        valid = [
            (img, self.world_points[i])
            for i, img in enumerate(self.image_points)
            if img
        ]
        ok = len(valid) >= 4
        if ok:
            A = np.hstack(
                [np.array([v[0] for v in valid]), np.ones((len(valid), 1))]
            )
            B = np.hstack(
                [np.array([v[1] for v in valid]), np.ones((len(valid), 1))]
            )
            ok = (
                np.linalg.matrix_rank(A) >= 3
                and np.linalg.matrix_rank(B) >= 3
                and len({tuple(p) for _, p in valid}) == len(valid)
            )
        self.apply_button.set_sensitive(ok)

    def on_apply_clicked(self, _):
        pts, wpts = [], []
        for i, img in enumerate(self.image_points):
            if not img:
                continue
            wx, wy = (
                self.bubble.get_world_coords()
                if i == self.active_point_index
                else self.world_points[i]
            )
            pts.append(img)
            wpts.append((wx, wy))
        if len(pts) < 4:
            raise ValueError("Less than 4 points.")
        self.camera.image_to_world = (pts, wpts)
        logger.info("Camera alignment applied.")
        self.close()

    def on_cancel_clicked(self, _):
        self.camera_display.stop()
        self.close()

    def _display_to_image_coords(self, dx, dy) -> Tuple[float, float]:
        dw, dh = (
            self.camera_display.get_width(),
            self.camera_display.get_height(),
        )
        iw, ih = self.camera.resolution
        return (
            dx / (dw / iw if dw > 0 else 1),
            (dh - dy) / (dh / ih if dh > 0 else 1),
        )

    def _find_point_near(self, x, y, threshold=10) -> int:
        for i, pt in enumerate(self.image_points):
            if pt and math.hypot(pt[0] - x, pt[1] - y) < threshold:
                return i
        return -1
