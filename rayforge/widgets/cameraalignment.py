import logging
from typing import List, Optional, Tuple
import numpy as np
from blinker import Signal
from gi.repository import Gtk, Adw, Gdk
from ..models.camera import Camera, Pos
from .cameradisplay import CameraDisplay


logger = logging.getLogger(__name__)


class CameraAlignmentPointRow(Adw.ActionRow):
    def __init__(self, point_index: int, **kwargs):
        super().__init__(**kwargs)
        self.point_index = point_index
        self.set_title(f"Point {point_index + 1}")
        self.set_subtitle("")  # Initially not active
        self.set_activatable(True)

        # Define blinker signals
        self.row_activated = Signal()
        self.value_changed = Signal()
        self.delete_requested = Signal()

        # Replace Entries with SpinButtons for World X and Y
        adjustment_x = Gtk.Adjustment.new(
            0.0, -10000.0, 10000.0, 0.1, 1.0, 0.0
        )
        self.world_x_spin = Gtk.SpinButton.new(adjustment_x, 0.1, 2)
        self.world_x_spin.set_valign(Gtk.Align.CENTER)
        adjustment_y = Gtk.Adjustment.new(
            0.0, -10000.0, 10000.0, 0.1, 1.0, 0.0
        )
        self.world_y_spin = Gtk.SpinButton.new(adjustment_y, 0.1, 2)
        self.world_y_spin.set_valign(Gtk.Align.CENTER)

        self.input_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6
        )
        self.input_box.append(self.world_x_spin)
        self.input_box.append(self.world_y_spin)
        self.add_suffix(self.input_box)

        # Add delete button
        self.delete_button = Gtk.Button.new_from_icon_name(
            "edit-delete-symbolic"
        )
        self.delete_button.set_valign(Gtk.Align.CENTER)
        self.delete_button.set_tooltip_text("Delete this point")
        self.add_suffix(self.delete_button)
        self.delete_button.connect("clicked", self.on_delete_clicked)

        self.is_active = False
        self.image_x = None
        self.image_y = None

        # Use separate EventControllerFocus for each spinbox
        focus_controller_x = Gtk.EventControllerFocus()
        focus_controller_x.connect(
            "enter", self.on_spin_focus, self.world_x_spin
        )
        self.world_x_spin.add_controller(focus_controller_x)

        focus_controller_y = Gtk.EventControllerFocus()
        focus_controller_y.connect(
            "enter", self.on_spin_focus, self.world_y_spin
        )
        self.world_y_spin.add_controller(focus_controller_y)

        # Connect value-changed to emit signal
        self.world_x_spin.connect("value-changed", self.on_value_changed)
        self.world_y_spin.connect("value-changed", self.on_value_changed)

    def on_spin_focus(self, controller, widget):
        self.row_activated.send(self, widget=widget)

    def on_value_changed(self, widget):
        self.value_changed.send(self)

    def on_delete_clicked(self, button):
        self.delete_requested.send(self)

    def set_active(self, active: bool):
        self.is_active = active
        if active:
            self.set_subtitle("Click a position in the image")
        else:
            self.set_subtitle("")

    def set_image_coords(self, x: float, y: float):
        self.image_x = x
        self.image_y = y
        self.set_title(
            f"Point {self.point_index + 1} (X: {x:.2f}, Y: {y:.2f})"
        )

    def get_image_coords(self) -> Optional[Tuple[float, float]]:
        if self.image_x is not None and self.image_y is not None:
            return (self.image_x, self.image_y)
        return None

    def get_world_coords(self) -> Tuple[float, float]:
        x = self.world_x_spin.get_value()
        y = self.world_y_spin.get_value()
        return x, y

    def clear_focus(self):
        if self.world_x_spin.has_focus() or self.world_y_spin.has_focus():
            window = self.world_x_spin.get_ancestor(Gtk.Window)
            if window:
                window.set_focus(None)


class CameraAlignmentDialog(Adw.MessageDialog):
    def __init__(self, parent: Gtk.Window, camera: Camera, **kwargs):
        super().__init__(
            transient_for=parent,
            modal=True,
            heading=f"{camera.name} - Image Alignment",
            close_response="cancel",
            **kwargs
        )
        self.camera = camera
        self.set_title(f"{self.camera.name} - Image Alignment")
        self.image_points: List[Pos] = []
        self.world_points: List[Pos] = []
        self.point_rows: List[CameraAlignmentPointRow] = []
        self.active_point_row: Optional[CameraAlignmentPointRow] = None

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.set_extra_child(main_box)

        # Camera Display
        self.camera_display = CameraDisplay(self.camera)
        self.camera_display.set_size_request(1280, 960)
        main_box.append(self.camera_display)

        # Event handling for point selection
        self.gesture_click = Gtk.GestureClick.new()
        self.gesture_click.connect("pressed", self.on_image_click)
        self.camera_display.add_controller(self.gesture_click)

        # Points List
        points_group = Adw.PreferencesGroup(title="Alignment Points")
        main_box.append(points_group)
        self.points_group = points_group

        # Buttons
        button_box = Gtk.Box(
            spacing=10, orientation=Gtk.Orientation.HORIZONTAL
        )
        main_box.append(button_box)

        add_point_button = Gtk.Button(label="Add Point")
        add_point_button.connect("clicked", self.on_add_point_clicked)
        button_box.append(add_point_button)

        clear_points_button = Gtk.Button(label="Clear Points")
        clear_points_button.connect("clicked", self.on_clear_points_clicked)
        button_box.append(clear_points_button)

        self.add_response("cancel", "Cancel")
        self.add_response("apply", "Apply")
        self.set_default_response("apply")
        self.connect("response", self.on_dialog_response)

        # Populate with existing points or add four initial rows
        if self.camera.image_to_world:
            image_points, world_points = self.camera.image_to_world
            for img_pt, world_pt in zip(image_points, world_points):
                new_row = CameraAlignmentPointRow(len(self.point_rows))
                new_row.connect("activated", self.set_active_row)
                new_row.row_activated.connect(self.set_active_row)
                new_row.value_changed.connect(
                    self.update_apply_button_sensitivity
                )
                new_row.delete_requested.connect(self.on_row_delete_requested)
                new_row.set_image_coords(img_pt[0], img_pt[1])
                new_row.world_x_spin.set_value(world_pt[0])
                new_row.world_y_spin.set_value(world_pt[1])
                self.image_points.append(img_pt)
                self.world_points.append(world_pt)
                self.point_rows.append(new_row)
                self.points_group.add(new_row)
        else:
            # Add four initial rows
            for _ in range(4):
                self.on_add_point_clicked(None)
        self.set_active_row(self.point_rows[0])
        self.update_delete_buttons()
        self.update_apply_button_sensitivity()

    def set_active_row(self, row: CameraAlignmentPointRow, widget=None):
        # Clear focus from all rows except the new one
        for r in self.point_rows:
            if r != row:
                r.remove_css_class("active-row")
                r.set_active(False)
                r.clear_focus()

        self.active_point_row = row
        self.active_point_row.add_css_class("active-row")
        self.active_point_row.set_active(True)

        # Set focus to the specific widget if provided,
        # otherwise to world_x_spin
        if widget and widget in (row.world_x_spin, row.world_y_spin):
            widget.grab_focus()
        else:
            row.world_x_spin.grab_focus()

        # Update camera display with active point index
        self.camera_display.set_marked_points(
            self.image_points, self.active_point_row.point_index
        )

    def on_image_click(self, gesture, n_press, x, y):
        if (
            gesture.get_current_button() == Gdk.BUTTON_PRIMARY
            and self.active_point_row
        ):
            # Account for CameraDisplay scaling and invert Y-axis:
            # (0,0) at bottom-left
            display_width, display_height = (
                self.camera_display.get_size_request()
            )
            img_width, img_height = self.camera.resolution
            scale_x = img_width / display_width
            scale_y = img_height / display_height
            x_scaled = x * scale_x
            y_scaled = (display_height - y) * scale_y

            # Check if click is near a marked point
            for i, (img_x, img_y) in enumerate(self.image_points):
                distance = ((img_x - x_scaled)**2 + (img_y - y_scaled)**2)**0.5
                if distance < 10:
                    self.set_active_row(self.point_rows[i])
                    return

            # If not near a marked point, set new point for active row
            if self.active_point_row:
                self.active_point_row.set_image_coords(x_scaled, y_scaled)
                idx = self.active_point_row.point_index
                if idx < len(self.image_points):
                    self.image_points[idx] = (x_scaled, y_scaled)
                else:
                    self.image_points.append((x_scaled, y_scaled))
                self.camera_display.set_marked_points(
                    self.image_points, self.active_point_row.point_index
                )
                self.update_apply_button_sensitivity()

    def on_add_point_clicked(self, button):
        new_row = CameraAlignmentPointRow(len(self.point_rows))
        new_row.connect("activated", self.set_active_row)
        new_row.row_activated.connect(self.set_active_row)
        new_row.value_changed.connect(self.update_apply_button_sensitivity)
        new_row.delete_requested.connect(self.on_row_delete_requested)
        self.point_rows.append(new_row)
        self.points_group.add(new_row)
        self.set_active_row(new_row)
        self.update_delete_buttons()
        self.update_apply_button_sensitivity()

    def on_row_delete_requested(self, row):
        if len(self.point_rows) > 4:
            self.points_group.remove(row)
            self.point_rows.remove(row)
            self.image_points.pop(row.point_index)
            for i, r in enumerate(self.point_rows):
                r.point_index = i
                img_coords = r.get_image_coords()
                if img_coords:
                    r.set_image_coords(img_coords[0], img_coords[1])
                else:
                    r.set_title(f"Point {i + 1}")
            self.camera_display.set_marked_points(
                self.image_points,
                self.active_point_row.point_index
                if self.active_point_row
                else -1,
            )
            self.update_delete_buttons()
            self.update_apply_button_sensitivity()
        else:
            self.show_alert_dialog(
                "Cannot Delete", "At least 4 points are required."
            )

    def update_delete_buttons(self):
        for row in self.point_rows:
            row.delete_button.set_sensitive(len(self.point_rows) > 4)

    def on_clear_points_clicked(self, button):
        self.image_points = []
        self.world_points = []
        self.active_point_row = None
        for row in self.point_rows:
            self.points_group.remove(row)
        self.point_rows = []

        # Add four new rows
        for _ in range(4):
            self.on_add_point_clicked(None)
        self.camera_display.set_marked_points(self.image_points, -1)
        self.update_apply_button_sensitivity()

    def update_apply_button_sensitivity(self, *args):
        # Collect valid points
        valid_image_points = []
        valid_world_points = []
        for row in self.point_rows:
            img_coords = row.get_image_coords()
            if img_coords is None:
                continue
            img_x, img_y = img_coords
            world_x, world_y = row.get_world_coords()
            valid_image_points.append([img_x, img_y])
            valid_world_points.append([world_x, world_y])

        # Check for at least 4 points and non-degeneracy
        valid = len(valid_image_points) >= 4 and len(valid_world_points) >= 4
        if valid:
            try:
                # Check if points are non-degenerate (not collinear/coincident)
                img_matrix = np.array(valid_image_points, dtype=np.float32)
                world_matrix = np.array(valid_world_points, dtype=np.float32)
                # Add homogeneous coordinate for rank check
                img_matrix_h = np.hstack(
                    [img_matrix, np.ones((img_matrix.shape[0], 1))]
                )
                world_matrix_h = np.hstack(
                    [world_matrix, np.ones((world_matrix.shape[0], 1))]
                )
                # Rank of at least 3 indicates points are not collinear
                img_rank = np.linalg.matrix_rank(img_matrix_h)
                world_rank = np.linalg.matrix_rank(world_matrix_h)
                valid = img_rank >= 3 and world_rank >= 3
                # Check for duplicate world points
                if valid:
                    world_points_set = {tuple(pt) for pt in valid_world_points}
                    valid = len(world_points_set) == len(valid_world_points)
                    if not valid:
                        logger.debug(
                            "Invalid point configuration: "
                            "duplicate world points"
                        )
            except np.linalg.LinAlgError:
                valid = False
                logger.debug("Invalid point configuration: singular matrix")

        # Enable "Apply" button only if valid
        self.set_response_enabled("apply", valid)

    def on_dialog_response(self, dialog, response_id):
        if response_id == "apply":
            valid_image_points = []
            valid_world_points = []
            for i, row in enumerate(self.point_rows):
                # Check if image coordinates are set
                img_coords = row.get_image_coords()
                if img_coords is None:
                    continue

                try:
                    img_x, img_y = img_coords
                    world_x, world_y = row.get_world_coords()
                    valid_image_points.append((img_x, img_y))
                    valid_world_points.append((world_x, world_y))
                except ValueError:
                    logger.warning(
                        f"Skipping row {i} due to invalid coordinate input."
                    )
                    continue

            if len(valid_image_points) < 4 or len(valid_world_points) < 4:
                raise ValueError(
                    "Attempted to apply alignment with less than 4 points."
                )

            try:
                self.camera.image_to_world = (
                    valid_image_points, valid_world_points
                )
                logger.info("Camera alignment applied.")
                self.close()
            except ValueError as e:
                logger.error(f"Error setting corresponding points: {e}")
                self.show_alert_dialog("Alignment Error", str(e))
            except Exception as e:
                logger.error(f"Unexpected error during alignment: {e}")
                self.show_alert_dialog(
                    "Error", f"An unexpected error occurred: {e}"
                )
        elif response_id == "cancel":
            logger.debug(
                "CameraAlignmentDialog closing, calling "
                f"CameraDisplay.stop() for camera {self.camera.name}"
            )
            self.camera_display.stop()
            self.close()

    def show_alert_dialog(self, title: str, message: str):
        dialog = Adw.AlertDialog.new(title, message)
        dialog.add_response("ok", "OK")
        dialog.set_default_response("ok")
        dialog.set_can_close(True)
        dialog.present(self.get_ancestor(Gtk.Window))
