import logging
import math
from typing import List, Optional, Tuple
import numpy as np
from gi.repository import Gtk, Adw, Gdk, GLib  # type: ignore
from ..models.camera import Camera, Pos
from .display_widget import CameraDisplay
from .point_bubble_widget import PointBubbleWidget

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

        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(
            b"""
            .info-highlight {
                background-color: @accent_bg_color;
                color: @accent_fg_color;
                border-radius: 6px;
                padding: 8px 12px;
            }
            """
        )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(content)

        header_bar = Adw.HeaderBar()
        header_bar.set_title_widget(
            Adw.WindowTitle(
                title=_("{camera_name} – Image Alignment").format(
                    camera_name=camera.name
                ),
                subtitle="",
            )
        )
        content.append(header_bar)

        vbox = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            margin_top=24,
            margin_bottom=24,
            margin_start=24,
            margin_end=24,
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

        # A floating, dismissible info box
        self.info_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            margin_top=24,  # Increased top margin
            margin_start=12,
            margin_end=12,
        )
        self.info_box.add_css_class("info-highlight")
        self.info_box.set_valign(Gtk.Align.START)
        self.info_box.set_halign(Gtk.Align.CENTER)
        self.overlay.add_overlay(self.info_box)

        icon = Gtk.Image.new_from_icon_name("dialog-info-symbolic")
        icon.set_valign(Gtk.Align.CENTER)  # Vertically centered
        self.info_box.append(icon)

        info_label = Gtk.Label(
            label=_(
                "Click the image to add new reference points. "
                "Click or drag existing points to edit them."
            ),
            xalign=0,
        )
        info_label.set_wrap(True)
        info_label.set_hexpand(True)
        self.info_box.append(info_label)

        dismiss_button = Gtk.Button.new_from_icon_name("window-close-symbolic")
        dismiss_button.add_css_class("flat")
        dismiss_button.set_valign(Gtk.Align.CENTER)  # Vertically centered
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

        self.camera_display.connect("realize", self._on_display_ready)
        self.camera_display.connect(
            "resize", lambda w, x, y: self._on_display_ready()
        )

        if camera.image_to_world:
            img_pts, wld_pts = camera.image_to_world
            self.image_points, self.world_points = list(img_pts), list(wld_pts)

        self.set_active_point(0)
        self.update_apply_button_sensitivity()

    def _on_display_ready(self, *args):
        if not self._display_ready:
            self._display_ready = True
            GLib.idle_add(self._position_bubble)
        else:
            self._position_bubble()

    def _position_bubble(self) -> bool:
        # Wait until the display is ready and a point is selected.
        if not self._display_ready or self.active_point_index < 0:
            return False

        # Get the coordinates of the active point.
        coords = self.image_points[self.active_point_index]
        if coords is None:
            return False
        img_x, img_y = coords

        # Get the dimensions of the camera display.
        display_width, display_height = (
            self.camera_display.get_width(),
            self.camera_display.get_height(),
        )
        if display_width <= 0 or display_height <= 0:
            return True  # Try again if the display is not ready.

        # Convert image coordinates to display coordinates.
        source_width, source_height = self.camera.resolution
        display_x = img_x * (display_width / source_width)
        display_y = display_height - (img_y * (display_height / source_height))

        # Get the dimensions of the bubble widget.
        alloc = self.bubble.get_allocation()
        bubble_width, bubble_height = alloc.width, alloc.height
        if bubble_width <= 0 or bubble_height <= 0:
            return True  # Try again if the bubble is not ready.

        # Center the bubble horizontally on the point, but keep it inside
        # the display area.
        x = max(
            0, min(display_x - bubble_width / 2, display_width - bubble_width)
        )

        # Position the bubble below the point.
        y = display_y + 10
        # If it goes off-screen, position it above the point.
        if y + bubble_height > display_height:
            y = max(0, display_y - bubble_height - 10)

        # Set the position of the bubble.
        self.bubble.set_margin_start(int(x))
        self.bubble.set_margin_top(int(y))

        # Make the bubble visible if it's not already.
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

    def on_drag_begin(self, gesture, x, y):
        image_x, image_y = self._display_to_image_coords(x, y)
        point_index = self._find_point_near(image_x, image_y)
        if point_index >= 0:
            point = self.image_points[point_index]
            if point is None:
                self.dragging_point_index = -1
                gesture.set_state(Gtk.EventSequenceState.DENIED)
                return

            self.dragging_point_index = point_index
            self.drag_start_image_x, self.drag_start_image_y = point
            gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        else:
            self.dragging_point_index = -1
            gesture.set_state(Gtk.EventSequenceState.DENIED)

    def on_drag_update(self, gesture, dx, dy):
        idx = self.dragging_point_index
        if idx < 0:
            return

        display_width, display_height = (
            self.camera_display.get_width(),
            self.camera_display.get_height(),
        )
        image_width, image_height = self.camera.resolution

        scale_x = display_width / image_width if display_width > 0 else 1
        scale_y = display_height / image_height if display_height > 0 else 1

        new_image_x = self.drag_start_image_x + dx / scale_x
        new_image_y = self.drag_start_image_y - dy / scale_y

        self.image_points[idx] = new_image_x, new_image_y
        if idx == self.active_point_index:
            self.bubble.set_image_coords(new_image_x, new_image_y)
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
        # Update the world coordinates of the active point from the bubble.
        if self.active_point_index >= 0 and self.active_point_index < len(
            self.world_points
        ):
            self.world_points[self.active_point_index] = (
                self.bubble.get_world_coords()
            )

        # Get a list of all valid points (i.e., points that have been set).
        valid_points = [
            (img, self.world_points[i])
            for i, img in enumerate(self.image_points)
            if img
        ]

        # We need at least 4 points for a valid transformation.
        can_apply = len(valid_points) >= 4
        if can_apply:
            # Check for collinearity of points. For a valid perspective
            # transform, we need at least 3 non-collinear points.
            # The rank of the matrix of homogeneous coordinates will be 3
            # if they are not collinear.
            image_coords = np.array([p[0] for p in valid_points])
            world_coords = np.array([p[1] for p in valid_points])

            image_points_matrix = np.hstack(
                [image_coords, np.ones((len(valid_points), 1))]
            )
            world_points_matrix = np.hstack(
                [world_coords, np.ones((len(valid_points), 1))]
            )

            # Also check that world points are unique.
            world_points_are_unique = len(
                {tuple(p) for p in world_coords}
            ) == len(world_coords)

            can_apply = (
                np.linalg.matrix_rank(image_points_matrix) >= 3
                and np.linalg.matrix_rank(world_points_matrix) >= 3
                and world_points_are_unique
            )

        # Enable or disable the "Apply" button based on the validity of the
        # points.
        self.apply_button.set_sensitive(can_apply)

    def on_apply_clicked(self, _):
        # Collect all valid points.
        image_points = []
        world_points = []
        for i, img_coords in enumerate(self.image_points):
            if not img_coords:
                continue

            # Get the world coordinates from the bubble if it's the active
            # point.
            world_x, world_y = (
                self.bubble.get_world_coords()
                if i == self.active_point_index
                else self.world_points[i]
            )
            image_points.append(img_coords)
            world_points.append((world_x, world_y))

        # Ensure we have enough points for the transformation.
        if len(image_points) < 4:
            raise ValueError("Less than 4 points for alignment.")

        # Apply the new alignment to the camera.
        self.camera.image_to_world = (image_points, world_points)
        logger.info("Camera alignment applied.")
        self.close()

    def on_cancel_clicked(self, _):
        self.camera_display.stop()
        self.close()

    def _display_to_image_coords(
        self, display_x: float, display_y: float
    ) -> Tuple[float, float]:
        """Converts display coordinates to image coordinates."""
        display_width, display_height = (
            self.camera_display.get_width(),
            self.camera_display.get_height(),
        )
        image_width, image_height = self.camera.resolution

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
