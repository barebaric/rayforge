import logging
import math
from typing import List, Optional, Tuple
from gettext import gettext as _

import numpy as np
from gi.repository import Adw, Gdk, GLib, Gtk, Graphene

from ...camera.controller import CameraController
from ...camera.models.camera import Pos
from ..icons import get_icon
from ..shared.gtk import apply_css
from ..shared.patched_dialog_window import PatchedDialogWindow
from ..canvas.worldsurface import WorldSurface
from .point_bubble_widget import PointBubbleWidget

logger = logging.getLogger(__name__)


class CameraAlignmentSurface(WorldSurface):
    def __init__(self, dialog, controller: CameraController, **kwargs):
        w, h = controller.resolution
        super().__init__(
            width_mm=w, height_mm=h, show_grid=False, show_axis=False, **kwargs
        )
        self.dialog = dialog
        self.controller = controller

        self.controller.subscribe()
        self.controller.image_captured.connect(self._on_image_captured)

        self.dragging_point_index = -1
        self.drag_offset_x = 0.0
        self.drag_offset_y = 0.0

        click = Gtk.GestureClick.new()
        click.set_button(Gdk.BUTTON_PRIMARY)
        click.connect("pressed", self.on_image_click)
        self.add_controller(click)

        drag = Gtk.GestureDrag.new()
        drag.set_button(Gdk.BUTTON_PRIMARY)
        drag.connect("drag-begin", self.on_drag_begin)
        drag.connect("drag-update", self.on_drag_update)
        drag.connect("drag-end", self.on_drag_end)
        self.add_controller(drag)

    def stop(self):
        self.controller.unsubscribe()

    def _on_image_captured(self, _):
        w, h = self.controller.resolution
        if w != self.width_mm or h != self.height_mm:
            self.set_size(w, h)
        self.queue_draw()

    def get_image_coords(self, x, y):
        widget_w, widget_h = self.get_width(), self.get_height()
        if widget_w <= 0 or widget_h <= 0:
            return 0.0, 0.0
        content_x, content_y, content_w, content_h = (
            self._axis_renderer.get_content_layout(widget_w, widget_h)
        )

        scale_x = content_w / self.width_mm if self.width_mm > 0 else 1
        scale_y = content_h / self.height_mm if self.height_mm > 0 else 1

        vx = x - content_x
        vy = y - content_y
        vx /= self.zoom_level
        vy /= self.zoom_level
        vx /= scale_x
        vy = (content_h - vy) / scale_y

        world_x = vx + self.pan_x_mm
        world_y = vy + self.pan_y_mm

        image_x = world_x
        image_y = world_y
        return image_x, image_y

    def _find_point_near(self, x, y, threshold=10):
        widget_w, widget_h = self.get_width(), self.get_height()
        if widget_w <= 0 or widget_h <= 0:
            return -1
        content_x, content_y, content_w, content_h = (
            self._axis_renderer.get_content_layout(widget_w, widget_h)
        )
        scale_x = content_w / self.width_mm if self.width_mm > 0 else 1

        scaled_threshold = threshold / (self.zoom_level * scale_x)

        for i, pt in enumerate(self.dialog.image_points or []):
            if pt is not None:
                if math.hypot(pt[0] - x, pt[1] - y) < scaled_threshold:
                    return i
        return -1

    def on_image_click(self, gesture, n, x, y):
        image_x, image_y = self.get_image_coords(x, y)
        point_index = self._find_point_near(image_x, image_y)

        if point_index >= 0:
            self.dialog.set_active_point(point_index)
        else:
            self.dialog.image_points.append((image_x, image_y))
            self.dialog.world_points.append((0.0, 0.0))
            self.dialog.set_active_point(len(self.dialog.image_points) - 1)
        self.queue_draw()
        self.dialog.update_apply_button_sensitivity()

    def on_drag_begin(self, gesture, start_x, start_y):
        self.dialog._interaction_in_progress = True
        image_x, image_y = self.get_image_coords(start_x, start_y)
        point_index = self._find_point_near(image_x, image_y)
        if point_index >= 0:
            self.dragging_point_index = point_index
            pt = self.dialog.image_points[point_index]
            if pt is not None:
                self.drag_offset_x = pt[0] - image_x
                self.drag_offset_y = pt[1] - image_y
            else:
                self.drag_offset_x = 0.0
                self.drag_offset_y = 0.0
            self.dialog.set_active_point(point_index)
            self.dialog._interaction_in_progress = True
            self.dialog.bubble.set_visible(True)
            gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        else:
            self.dragging_point_index = -1
            gesture.set_state(Gtk.EventSequenceState.DENIED)

    def on_drag_update(self, gesture, offset_x, offset_y):
        idx = self.dragging_point_index
        if idx < 0:
            return
        ok, start_x, start_y = gesture.get_start_point()
        if not ok:
            return
        current_x = start_x + offset_x
        current_y = start_y + offset_y
        image_x, image_y = self.get_image_coords(current_x, current_y)
        new_x = image_x + self.drag_offset_x
        new_y = image_y + self.drag_offset_y
        self.dialog.image_points[idx] = (new_x, new_y)

        if idx == self.dialog.active_point_index:
            self.dialog.bubble.set_image_coords(new_x, new_y)
            self.dialog._position_bubble()
        self.queue_draw()

    def on_drag_end(self, gesture, offset_x, offset_y):
        self.dialog._interaction_in_progress = False
        if self.dragging_point_index >= 0:
            self.dragging_point_index = -1
            self.dialog._position_bubble()

    def do_snapshot(self, snapshot: Gtk.Snapshot) -> None:
        width, height = self.get_width(), self.get_height()
        ctx = snapshot.append_cairo(Graphene.Rect().init(0, 0, width, height))

        content_x, content_y, content_w, content_h = (
            self._axis_renderer.get_content_layout(width, height)
        )
        scale_x = content_w / self.width_mm if self.width_mm > 0 else 1
        scale_y = content_h / self.height_mm if self.height_mm > 0 else 1

        ctx.save()
        ctx.translate(content_x, content_y)
        ctx.scale(self.zoom_level, self.zoom_level)
        ctx.translate(0, content_h)
        ctx.scale(scale_x, -scale_y)
        ctx.translate(-self.pan_x_mm, -self.pan_y_mm)

        pixbuf = self.controller.pixbuf
        if pixbuf:
            ctx.save()
            ctx.translate(0, self.height_mm)
            ctx.scale(1, -1)
            Gdk.cairo_set_source_pixbuf(ctx, pixbuf, 0, 0)
            ctx.paint()
            ctx.restore()

        for i, pt in enumerate(self.dialog.image_points):
            if pt is not None:
                world_x = pt[0]
                world_y = pt[1]
                radius = 5 / (self.zoom_level * scale_x)
                ctx.arc(world_x, world_y, radius, 0, 2 * 3.14159)
                if i == self.dialog.active_point_index:
                    ctx.set_source_rgba(1, 0.2, 0.2, 0.8)
                else:
                    ctx.set_source_rgba(0.2, 0.6, 1.0, 0.8)
                ctx.fill()
                ctx.set_source_rgba(1, 1, 1, 1)
                ctx.set_line_width(1.5 / (self.zoom_level * scale_x))
                ctx.stroke()

        ctx.restore()

        self._update_theme_colors()
        self._axis_renderer.draw_grid_and_labels(
            ctx, self.view_transform, width, height
        )


class CameraAlignmentDialog(PatchedDialogWindow):
    def __init__(
        self, parent: Gtk.Window, controller: CameraController, **kwargs
    ):
        super().__init__(
            transient_for=parent,
            modal=True,
            default_width=1280,
            default_height=960,
            **kwargs,
        )

        self.controller = controller
        self.camera = controller.config
        self.image_points: List[Optional[Pos]] = []
        self.world_points: List[Pos] = []

        self.active_point_index = -1
        self._display_ready = False

        # Interaction Lock: Blocks automatic idle positioning while dragging
        # manually
        self._interaction_in_progress = False

        apply_css(
            """
            .info-highlight {
                background-color: @accent_bg_color;
                color: @accent_fg_color;
                border-radius: 6px;
                padding: 8px 12px;
            }
            """
        )

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(content)

        # --- Header Bar with Zoom Controls ---
        header_bar = Adw.HeaderBar()
        header_title = _("{camera_name} – Image Alignment").format(
            camera_name=self.camera.name
        )
        header_bar.set_title_widget(
            Adw.WindowTitle(
                title=header_title,
                subtitle="",
            )
        )
        content.append(header_bar)

        zoom_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        zoom_box.add_css_class("linked")

        btn_zoom_out = Gtk.Button(
            child=get_icon("zoom-out-symbolic"),
            tooltip_text=_("Zoom Out (Scroll Down)"),
        )
        btn_zoom_out.connect("clicked", self.on_zoom_out_click)

        btn_zoom_fit = Gtk.Button(
            child=get_icon("zoom-fit-best-symbolic"),
            tooltip_text=_("Fit to Window"),
        )
        btn_zoom_fit.connect("clicked", self.on_zoom_fit_click)

        btn_zoom_in = Gtk.Button(
            child=get_icon("zoom-in-symbolic"),
            tooltip_text=_("Zoom In (Scroll Up)"),
        )
        btn_zoom_in.connect("clicked", self.on_zoom_in_click)

        zoom_box.append(btn_zoom_out)
        zoom_box.append(btn_zoom_fit)
        zoom_box.append(btn_zoom_in)
        header_bar.pack_start(zoom_box)

        vbox = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            margin_top=24,
            margin_bottom=24,
            margin_start=24,
            margin_end=24,
        )
        content.append(vbox)

        # --- Viewport & Hierarchy Setup ---
        self.main_overlay = Gtk.Overlay()
        vbox.append(self.main_overlay)

        self.camera_display = CameraAlignmentSurface(self, controller)
        self.camera_display.set_hexpand(True)
        self.camera_display.set_vexpand(True)
        self.main_overlay.set_child(self.camera_display)

        self.bubble = PointBubbleWidget(0)
        # Prevent bubble from forcing parent size or interacting with expand
        # logic
        self.bubble.set_hexpand(False)
        self.bubble.set_vexpand(False)
        self.main_overlay.add_overlay(self.bubble)
        self.bubble.set_halign(Gtk.Align.START)
        self.bubble.set_valign(Gtk.Align.START)
        self.bubble.set_visible(False)
        self.bubble.value_changed.connect(self.update_apply_button_sensitivity)
        self.bubble.delete_requested.connect(self.on_point_delete_requested)
        self.bubble.focus_requested.connect(self.on_bubble_focus_requested)
        self.bubble.nudge_requested.connect(self.on_nudge_requested)

        self.info_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            margin_top=24,
            margin_start=12,
            margin_end=12,
        )
        self.info_box.add_css_class("info-highlight")
        self.info_box.set_valign(Gtk.Align.START)
        self.info_box.set_halign(Gtk.Align.CENTER)
        self.main_overlay.add_overlay(self.info_box)

        icon = get_icon("info-symbolic")
        icon.set_valign(Gtk.Align.CENTER)
        self.info_box.append(icon)

        info_text = _(
            "Click the image to add reference points. Drag to move them.\n"
            "Scroll to Zoom. Middle-click and drag to Pan.\n"
            "Use the Arrow Keys to nudge the active point precisely."
        )
        info_label = Gtk.Label(label=info_text, xalign=0)
        info_label.set_wrap(True)
        info_label.set_hexpand(True)
        self.info_box.append(info_label)

        dismiss_button = Gtk.Button(child=get_icon("close-symbolic"))
        dismiss_button.add_css_class("flat")
        dismiss_button.set_valign(Gtk.Align.CENTER)
        dismiss_button.connect("clicked", lambda btn: self.info_box.hide())
        self.info_box.append(dismiss_button)

        btn_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            halign=Gtk.Align.END,
            margin_top=12,
        )
        vbox.append(btn_box)

        for label, cb in [
            (_("Reset Points"), self.on_reset_points_clicked),
            (_("Clear All Points"), self.on_clear_all_points_clicked),
            (_("Cancel"), self.on_cancel_clicked),
        ]:
            btn = Gtk.Button(label=label)
            btn.add_css_class("flat")
            btn.connect("clicked", cb)
            btn_box.append(btn)

        self.apply_button = Gtk.Button(label=_("Apply"))
        self.apply_button.add_css_class("suggested-action")
        self.apply_button.connect("clicked", self.on_apply_clicked)
        btn_box.append(self.apply_button)

        # --- Event Controllers ---
        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self.on_key_pressed)
        self.add_controller(key_controller)

        # --- Signal Connections ---
        self.camera_display.connect("realize", self._on_display_ready)

        if self.camera.image_to_world:
            img_pts, wld_pts = self.camera.image_to_world
            self.image_points, self.world_points = list(img_pts), list(wld_pts)

        self.set_active_point(0)
        self.update_apply_button_sensitivity()

    # --- Calculation Helpers ---

    def _calculate_bubble_margins(
        self, img_x: float, img_y: float
    ) -> Tuple[bool, int, int]:
        surface = self.camera_display
        widget_w, widget_h = surface.get_width(), surface.get_height()
        if widget_w <= 0 or widget_h <= 0:
            return False, 0, 0

        content_x, content_y, content_w, content_h = (
            surface._axis_renderer.get_content_layout(widget_w, widget_h)
        )

        scale_x = content_w / surface.width_mm if surface.width_mm > 0 else 1
        scale_y = content_h / surface.height_mm if surface.height_mm > 0 else 1

        world_x = img_x
        world_y = img_y

        vx = (world_x - surface.pan_x_mm) * scale_x
        vy = (world_y - surface.pan_y_mm) * scale_y
        vy = content_h - vy

        vx *= surface.zoom_level
        vy *= surface.zoom_level

        display_x = vx + content_x
        display_y = vy + content_y

        alloc = self.bubble.get_allocation()
        bubble_width, bubble_height = alloc.width, alloc.height

        x = display_x - (bubble_width / 2)
        x = max(12, min(x, widget_w - bubble_width - 12))

        y = display_y + 16
        if y + bubble_height > widget_h - 12:
            y = display_y - bubble_height - 16

        return True, int(x), int(y)

    # --- Zoom Logic ---

    def on_zoom_in_click(self, _):
        new_val = min(10.0, self.camera_display.zoom_level * 1.25)
        self.camera_display.set_zoom(new_val)

    def on_zoom_out_click(self, _):
        new_val = max(0.1, self.camera_display.zoom_level / 1.25)
        self.camera_display.set_zoom(new_val)

    def on_zoom_fit_click(self, _):
        self.camera_display.reset_view()

    # --- Interaction Logic ---

    def _on_display_ready(self, *args):
        if not self._display_ready:
            self._display_ready = True
            GLib.idle_add(self._position_bubble)
        else:
            self._position_bubble()

    def _position_bubble(self) -> bool:
        """
        Positions the bubble widget.
        Returns GLib.SOURCE_REMOVE strictly to ensure idle loops terminate.
        """
        if self._interaction_in_progress:
            return GLib.SOURCE_REMOVE

        if not self._display_ready or self.active_point_index < 0:
            return GLib.SOURCE_REMOVE

        coords = self.image_points[self.active_point_index]
        if coords is None:
            return GLib.SOURCE_REMOVE

        visible, x, y = self._calculate_bubble_margins(coords[0], coords[1])
        if not visible:
            return GLib.SOURCE_REMOVE

        # Optimization: Only update margins if changed to prevent thrashing
        if self.bubble.get_margin_start() != x:
            self.bubble.set_margin_start(x)
        if self.bubble.get_margin_top() != y:
            self.bubble.set_margin_top(y)

        if (
            not self.bubble.get_visible()
            and self.camera_display.dragging_point_index == -1
        ):
            self.bubble.set_visible(True)

        return GLib.SOURCE_REMOVE

    def set_active_point(self, index: int, widget=None):
        if index < 0 or index >= len(self.image_points):
            self.active_point_index = -1
            self.bubble.set_visible(False)
            self.camera_display.queue_draw()
            return

        self.active_point_index = index
        self.bubble.set_point_index(index)

        coords = self.image_points[index]
        if coords is not None:
            self.bubble.set_image_coords(*coords)

        self.bubble.set_world_coords(*self.world_points[index])

        if self.camera_display.dragging_point_index == -1:
            self._interaction_in_progress = False
            self._position_bubble()

        (widget or self.bubble.world_x_spin).grab_focus()
        self.camera_display.queue_draw()

    def on_bubble_focus_requested(self, bubble, widget):
        self.set_active_point(self.active_point_index, widget)

    def on_nudge_requested(self, bubble, dx, dy):
        if self.active_point_index < 0:
            return
        p = self.image_points[self.active_point_index]
        if p is None:
            return
        nx, ny = p[0] + dx, p[1] + dy
        self.image_points[self.active_point_index] = (nx, ny)
        self.bubble.set_image_coords(nx, ny)
        self._position_bubble()
        self.camera_display.queue_draw()

    def on_key_pressed(self, controller, keyval, keycode, state):
        if keyval == Gdk.KEY_Escape:
            self.close()
            return Gdk.EVENT_STOP

        focus_widget = self.get_focus()
        is_typing = isinstance(focus_widget, Gtk.Text) or isinstance(
            focus_widget, Gtk.SpinButton
        )

        if not is_typing and self.active_point_index >= 0:
            dx, dy = 0.0, 0.0
            step = 5.0 if (state & Gdk.ModifierType.SHIFT_MASK) else 0.5

            if keyval == Gdk.KEY_Up:
                dy = step
            elif keyval == Gdk.KEY_Down:
                dy = -step
            elif keyval == Gdk.KEY_Left:
                dx = -step
            elif keyval == Gdk.KEY_Right:
                dx = step
            elif keyval in (Gdk.KEY_Delete, Gdk.KEY_BackSpace):
                self.on_point_delete_requested(self.bubble)
                return Gdk.EVENT_STOP

            if dx != 0.0 or dy != 0.0:
                self.on_nudge_requested(self.bubble, dx, dy)
                return Gdk.EVENT_STOP

        return Gdk.EVENT_PROPAGATE

    def on_reset_points_clicked(self, _):
        self.image_points.clear()
        self.world_points.clear()
        if self.camera.image_to_world:
            image_points_data, world_points_data = self.camera.image_to_world
            self.image_points, self.world_points = (
                list(image_points_data),
                list(world_points_data),
            )
        else:
            self.image_points = [None] * 4
            self.world_points = [(0.0, 0.0)] * 4
        self.set_active_point(0)
        self.camera_display.queue_draw()
        self.update_apply_button_sensitivity()

    def on_clear_all_points_clicked(self, _):
        self.image_points.clear()
        self.world_points.clear()
        self.set_active_point(-1)
        self.camera_display.queue_draw()
        self.update_apply_button_sensitivity()

    def on_point_delete_requested(self, bubble):
        index = bubble.point_index
        if 0 <= index < len(self.image_points):
            self.image_points.pop(index)
            self.world_points.pop(index)
        if self.image_points:
            self.set_active_point(min(index, len(self.image_points) - 1))
        else:
            self.set_active_point(-1)

        self.camera_display.queue_draw()
        self.update_apply_button_sensitivity()

    def update_apply_button_sensitivity(self, *_):
        idx = self.active_point_index
        if idx >= 0 and idx < len(self.world_points):
            self.world_points[idx] = self.bubble.get_world_coords()

        valid_points = [
            (img, self.world_points[i])
            for i, img in enumerate(self.image_points or [])
            if img
        ]

        can_apply = len(valid_points) >= 4
        if can_apply:
            image_coords = np.array([p[0] for p in valid_points])
            world_coords = np.array([p[1] for p in valid_points])

            image_points_matrix = np.hstack(
                [image_coords, np.ones((len(valid_points), 1))]
            )
            world_points_matrix = np.hstack(
                [world_coords, np.ones((len(valid_points), 1))]
            )

            world_points_are_unique = len(
                {tuple(p) for p in world_coords}
            ) == len(world_coords)

            can_apply = (
                np.linalg.matrix_rank(image_points_matrix) >= 3
                and np.linalg.matrix_rank(world_points_matrix) >= 3
                and world_points_are_unique
            )

        self.apply_button.set_sensitive(can_apply)

    def on_apply_clicked(self, _):
        image_points = []
        world_points = []
        for i, img_coords in enumerate(self.image_points or []):
            if not img_coords:
                continue

            world_x, world_y = (
                self.bubble.get_world_coords()
                if i == self.active_point_index
                else self.world_points[i]
            )
            image_points.append(img_coords)
            world_points.append((world_x, world_y))

        if len(image_points) < 4:
            raise ValueError("Less than 4 points for alignment.")

        self.camera.image_to_world = (image_points, world_points)
        logger.info("Camera alignment applied.")
        self.close()

    def on_cancel_clicked(self, _):
        self.camera_display.stop()
        self.close()
