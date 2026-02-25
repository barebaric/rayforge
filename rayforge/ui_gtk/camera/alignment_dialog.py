import logging
import math
from typing import List, Optional, Tuple

import numpy as np
from gi.repository import Adw, Gdk, GLib, Gtk

from ...camera.controller import CameraController
from ...camera.models.camera import Pos
from ..icons import get_icon
from ..shared.gtk import apply_css
from ..shared.patched_dialog_window import PatchedDialogWindow
from .display_widget import CameraDisplay
from .point_bubble_widget import PointBubbleWidget

logger = logging.getLogger(__name__)


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
        self.dragging_point_index = -1

        # Drag State
        self.drag_start_display_x = 0.0
        self.drag_start_display_y = 0.0
        self.drag_offset_x = 0.0
        self.drag_offset_y = 0.0

        self._display_ready = False

        # Interaction Lock: Blocks automatic idle positioning while dragging
        # manually
        self._interaction_in_progress = False

        # Zoom & Pan State
        self._zoom_level = 1.0
        self._is_fitting = True
        self._pan_start_h = 0.0
        self._pan_start_v = 0.0

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
            icon_name="zoom-out-symbolic",
            tooltip_text=_("Zoom Out (Ctrl + Scroll Down)"))
        btn_zoom_out.connect("clicked", self.on_zoom_out_click)

        btn_zoom_fit = Gtk.Button(
            icon_name="zoom-fit-best-symbolic",
            tooltip_text=_("Fit to Window"))
        btn_zoom_fit.connect("clicked", self.on_zoom_fit_click)

        btn_zoom_in = Gtk.Button(
            icon_name="zoom-in-symbolic",
            tooltip_text=_("Zoom In (Ctrl + Scroll Up)"))
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

        self.scrolled_window = Gtk.ScrolledWindow()
        self.scrolled_window.set_hexpand(True)
        self.scrolled_window.set_vexpand(True)
        # Use AUTOMATIC so scrollbars appear only when zoomed in
        self.scrolled_window.set_policy(
            Gtk.PolicyType.AUTOMATIC,
            Gtk.PolicyType.AUTOMATIC)
        self.main_overlay.set_child(self.scrolled_window)

        self.scrolled_overlay = Gtk.Overlay()
        self.scrolled_overlay.set_hexpand(True)
        self.scrolled_overlay.set_vexpand(True)
        self.scrolled_window.set_child(self.scrolled_overlay)

        self.camera_display = CameraDisplay(controller)
        self.camera_display.set_halign(Gtk.Align.CENTER)
        self.camera_display.set_valign(Gtk.Align.CENTER)
        self.scrolled_overlay.set_child(self.camera_display)

        self.bubble = PointBubbleWidget(0)
        # Prevent bubble from forcing parent size or interacting with expand
        # logic
        self.bubble.set_hexpand(False)
        self.bubble.set_vexpand(False)
        self.scrolled_overlay.add_overlay(self.bubble)
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
            "Ctrl+Scroll to Zoom. Middle-click and drag to Pan.\n"
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
        click = Gtk.GestureClick.new()
        click.set_button(Gdk.BUTTON_PRIMARY)
        click.connect("pressed", self.on_image_click)
        self.camera_display.add_controller(click)

        # Left Click: Drag Points
        drag = Gtk.GestureDrag.new()
        drag.set_button(Gdk.BUTTON_PRIMARY)
        drag.connect("drag-begin", self.on_drag_begin)
        drag.connect("drag-update", self.on_drag_update)
        drag.connect("drag-end", self.on_drag_end)
        self.camera_display.add_controller(drag)

        # Middle Click: Pan
        pan_drag = Gtk.GestureDrag.new()
        pan_drag.set_button(Gdk.BUTTON_MIDDLE)
        pan_drag.connect("drag-begin", self.on_pan_begin)
        pan_drag.connect("drag-update", self.on_pan_update)
        pan_drag.connect("drag-end", self.on_pan_end)
        self.scrolled_overlay.add_controller(pan_drag)

        # Scroll: Zoom (Global Main Overlay)
        scroll = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL)
        scroll.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        scroll.connect("scroll", self.on_scroll)
        self.main_overlay.add_controller(scroll)

        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self.on_key_pressed)
        self.add_controller(key_controller)

        # --- Signal Connections ---
        self.camera_display.connect("realize", self._on_display_ready)
        self.scrolled_window.connect("notify::width", self._on_viewport_resize)
        self.scrolled_window.connect(
            "notify::height", self._on_viewport_resize)

        if self.camera.image_to_world:
            img_pts, wld_pts = self.camera.image_to_world
            self.image_points, self.world_points = list(img_pts), list(wld_pts)

        self.set_active_point(0)
        self.update_apply_button_sensitivity()

    # --- Calculation Helpers ---

    def _calculate_fit_zoom(self) -> float:
        viewport_w = self.scrolled_window.get_width()
        viewport_h = self.scrolled_window.get_height()
        image_w, image_h = self.controller.resolution

        if image_w <= 0 or image_h <= 0 or viewport_w <= 0 or viewport_h <= 0:
            return 1.0

        return min(viewport_w / image_w, viewport_h / image_h)

    def _calculate_bubble_margins(
            self, img_x: float, img_y: float) -> Tuple[bool, int, int]:
        display_width = self.camera_display.get_width()
        display_height = self.camera_display.get_height()

        if display_width <= 0 or display_height <= 0:
            return False, 0, 0

        source_width, source_height = self.controller.resolution
        display_x = img_x * (display_width / source_width)
        display_y = display_height - (img_y * (display_height / source_height))

        overlay_w = self.scrolled_overlay.get_width()
        overlay_h = self.scrolled_overlay.get_height()

        offset_x = max(0, (overlay_w - display_width) / 2)
        offset_y = max(0, (overlay_h - display_height) / 2)

        alloc = self.bubble.get_allocation()
        bubble_width, bubble_height = alloc.width, alloc.height

        x = offset_x + display_x - (bubble_width / 2)
        x = max(12, min(x, overlay_w - bubble_width - 12))

        y = offset_y + display_y + 16
        if y + bubble_height > overlay_h - 12:
            y = offset_y + display_y - bubble_height - 16

        return True, int(x), int(y)

    # --- Zoom Logic ---

    def _apply_zoom(self, new_zoom, center_x_ratio=0.5, center_y_ratio=0.5):
        old_zoom = self._zoom_level
        self._zoom_level = new_zoom

        image_w, image_h = self.controller.resolution
        hadj = self.scrolled_window.get_hadjustment()
        vadj = self.scrolled_window.get_vadjustment()

        viewport_w = self.scrolled_window.get_width()
        viewport_h = self.scrolled_window.get_height()

        visual_x = hadj.get_value() + (viewport_w * center_x_ratio)
        visual_y = vadj.get_value() + (viewport_h * center_y_ratio)

        absolute_x = visual_x / old_zoom if old_zoom > 0 else 0
        absolute_y = visual_y / old_zoom if old_zoom > 0 else 0

        if self._is_fitting:
            self.camera_display.set_size_request(-1, -1)
            self.camera_display.set_hexpand(True)
            self.camera_display.set_vexpand(True)
        else:
            self.camera_display.set_hexpand(False)
            self.camera_display.set_vexpand(False)
            req_w = int(image_w * self._zoom_level)
            req_h = int(image_h * self._zoom_level)
            self.camera_display.set_size_request(req_w, req_h)

        self.camera_display.queue_resize()

        def update_scroll():
            if not self._is_fitting:
                new_visual_x = absolute_x * self._zoom_level
                new_visual_y = absolute_y * self._zoom_level

                new_scroll_x = new_visual_x - (viewport_w * center_x_ratio)
                new_scroll_y = new_visual_y - (viewport_h * center_y_ratio)

                hadj.set_value(max(0, new_scroll_x))
                vadj.set_value(max(0, new_scroll_y))

            self._position_bubble()
            return GLib.SOURCE_REMOVE

        GLib.idle_add(update_scroll)

    def on_scroll(self, controller, dx, dy):
        state = controller.get_current_event_state()
        if state & Gdk.ModifierType.CONTROL_MASK:
            zoom_factor = 1.25 if dy < 0 else (1.0 / 1.25)

            event = controller.get_current_event()
            mx, my = 0, 0
            has_pointer = False

            if event:
                try:
                    pos = event.get_position()
                    if pos:
                        mx, my = pos
                        has_pointer = True
                except Exception:
                    pass

            viewport_w = self.scrolled_window.get_width()
            viewport_h = self.scrolled_window.get_height()

            if viewport_w <= 0 or viewport_h <= 0:
                return Gdk.EVENT_PROPAGATE

            ratio_x = 0.5
            ratio_y = 0.5

            if has_pointer:
                ratio_x = max(0.0, min(1.0, mx / viewport_w))
                ratio_y = max(0.0, min(1.0, my / viewport_h))

            if self._is_fitting:
                self._zoom_level = self._calculate_fit_zoom()
                self._is_fitting = False

            new_zoom = self._zoom_level * zoom_factor
            new_zoom = max(0.1, min(10.0, new_zoom))

            self._apply_zoom(new_zoom, ratio_x, ratio_y)
            return Gdk.EVENT_STOP

        return Gdk.EVENT_PROPAGATE

    def on_zoom_in_click(self, _):
        if self._is_fitting:
            self._zoom_level = self._calculate_fit_zoom()
            self._is_fitting = False
        new_val = min(10.0, self._zoom_level * 1.25)
        self._apply_zoom(new_val, 0.5, 0.5)

    def on_zoom_out_click(self, _):
        if self._is_fitting:
            self._zoom_level = self._calculate_fit_zoom()
            self._is_fitting = False
        new_val = max(0.1, self._zoom_level / 1.25)
        self._apply_zoom(new_val, 0.5, 0.5)

    def on_zoom_fit_click(self, _):
        self._is_fitting = True
        self._apply_zoom(1.0, 0.5, 0.5)

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

        if not self.bubble.get_visible() and self.dragging_point_index == -1:
            self.bubble.set_visible(True)

        return GLib.SOURCE_REMOVE

    def on_drag_begin(self, gesture, x, y):
        self._interaction_in_progress = True

        # 1. Calculate where the mouse is in image space
        mouse_image_x, mouse_image_y = self._display_to_image_coords(x, y)

        # 2. Find the point near this mouse location
        point_index = self._find_point_near(mouse_image_x, mouse_image_y)

        if point_index >= 0:
            self.dragging_point_index = point_index

            # 3. Store start coords
            self.drag_start_display_x = x
            self.drag_start_display_y = y

            # 4. Calculate offset
            pt_x, pt_y = self.image_points[point_index]
            self.drag_offset_x = pt_x - mouse_image_x
            self.drag_offset_y = pt_y - mouse_image_y

            # 5. Update UI state
            self.set_active_point(point_index)
            # Re-lock because set_active_point unlocks
            self._interaction_in_progress = True
            self.bubble.set_visible(True)

            gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        else:
            self.dragging_point_index = -1
            gesture.set_state(Gtk.EventSequenceState.DENIED)

    def on_drag_update(self, gesture, dx, dy):
        idx = self.dragging_point_index
        if idx < 0:
            return

        # 1. Calculate current mouse position in display pixels
        current_display_x = self.drag_start_display_x + dx
        current_display_y = self.drag_start_display_y + dy

        # 2. Convert to image space
        mouse_image_x, mouse_image_y = self._display_to_image_coords(
            current_display_x, current_display_y
        )

        # 3. Apply the original offset
        new_image_x = mouse_image_x + self.drag_offset_x
        new_image_y = mouse_image_y + self.drag_offset_y

        self.image_points[idx] = (new_image_x, new_image_y)

        if idx == self.active_point_index:
            self.bubble.set_image_coords(new_image_x, new_image_y)
            visible, mx, my = self._calculate_bubble_margins(
                new_image_x, new_image_y)
            if visible:
                if self.bubble.get_margin_start() != mx:
                    self.bubble.set_margin_start(mx)
                if self.bubble.get_margin_top() != my:
                    self.bubble.set_margin_top(my)

        self.camera_display.set_marked_points(
            self.image_points, self.active_point_index
        )
        self.camera_display.queue_draw()

    def on_drag_end(self, gesture, dx, dy):
        self._interaction_in_progress = False
        if self.dragging_point_index >= 0:
            self.dragging_point_index = -1
            self._position_bubble()

    def on_pan_begin(self, gesture, x, y):
        self._interaction_in_progress = True
        self._pan_start_h = self.scrolled_window.get_hadjustment().get_value()
        self._pan_start_v = self.scrolled_window.get_vadjustment().get_value()
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)

    def on_pan_update(self, gesture, dx, dy):
        h_adj = self.scrolled_window.get_hadjustment()
        v_adj = self.scrolled_window.get_vadjustment()
        h_adj.set_value(self._pan_start_h - dx)
        v_adj.set_value(self._pan_start_v - dy)

    def on_pan_end(self, gesture, dx, dy):
        self._interaction_in_progress = False
        self._position_bubble()

    def _on_viewport_resize(self, *args):
        if self._is_fitting:
            new_zoom = self._calculate_fit_zoom()
            if abs(new_zoom - self._zoom_level) > 0.001:
                self._apply_zoom(new_zoom, 0.5, 0.5)

    def on_image_click(self, gesture, n, x, y):
        if gesture.get_current_button() != Gdk.BUTTON_PRIMARY:
            return
        image_x, image_y = self._display_to_image_coords(x, y)
        point_index = self._find_point_near(image_x, image_y)

        if point_index >= 0:
            self.set_active_point(point_index)
        else:
            self.image_points.append((image_x, image_y))
            self.world_points.append((0.0, 0.0))
            self.set_active_point(len(self.image_points) - 1)

        self.camera_display.set_marked_points(
            self.image_points, self.active_point_index
        )
        self.update_apply_button_sensitivity()

    def set_active_point(self, index: int, widget=None):
        if index < 0 or index >= len(self.image_points):
            self.active_point_index = -1
            self.bubble.set_visible(False)
            self.camera_display.set_marked_points(self.image_points, -1)
            return

        self.active_point_index = index
        self.bubble.set_point_index(index)

        if self.image_points[index]:
            self.bubble.set_image_coords(*self.image_points[index])

        self.bubble.set_world_coords(*self.world_points[index])

        if self.dragging_point_index == -1:
            self._interaction_in_progress = False
            self._position_bubble()

        (widget or self.bubble.world_x_spin).grab_focus()
        self.camera_display.set_marked_points(self.image_points, index)

    def _display_to_image_coords(
        self, display_x: float, display_y: float
    ) -> Tuple[float, float]:
        display_width = self.camera_display.get_width()
        display_height = self.camera_display.get_height()
        image_width, image_height = self.controller.resolution

        if display_width <= 0 or display_height <= 0:
            return 0.0, 0.0

        scale_x = display_width / image_width
        scale_y = display_height / image_height

        image_x = display_x / scale_x
        image_y = (display_height - display_y) / scale_y
        return image_x, image_y

    def _find_point_near(self, x, y, threshold=10) -> int:
        for i, pt in enumerate(self.image_points):
            if pt and math.hypot(pt[0] - x, pt[1] - y) < threshold:
                return i
        return -1

    def on_bubble_focus_requested(self, bubble, widget):
        self.set_active_point(self.active_point_index, widget)

    def on_nudge_requested(self, bubble, dx, dy):
        if self.active_point_index < 0:
            return
        p = self.image_points[self.active_point_index]
        nx, ny = p[0] + dx, p[1] + dy
        self.image_points[self.active_point_index] = (nx, ny)
        self.bubble.set_image_coords(nx, ny)
        self._position_bubble()
        self.camera_display.set_marked_points(
            self.image_points, self.active_point_index)
        self.camera_display.queue_draw()

    def on_key_pressed(self, controller, keyval, keycode, state):
        if keyval == Gdk.KEY_Escape:
            self.close()
            return Gdk.EVENT_STOP

        focus_widget = self.get_focus()
        is_typing = isinstance(
            focus_widget,
            Gtk.Text) or isinstance(
            focus_widget,
            Gtk.SpinButton)

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
        self.update_apply_button_sensitivity()

    def on_clear_all_points_clicked(self, _):
        self.image_points.clear()
        self.world_points.clear()
        self.set_active_point(-1)
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

        self.camera_display.set_marked_points(
            self.image_points, self.active_point_index
        )
        self.update_apply_button_sensitivity()

    def update_apply_button_sensitivity(self, *_):
        idx = self.active_point_index
        if idx >= 0 and idx < len(self.world_points):
            self.world_points[idx] = self.bubble.get_world_coords()

        valid_points = [
            (img, self.world_points[i])
            for i, img in enumerate(self.image_points)
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
        for i, img_coords in enumerate(self.image_points):
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
