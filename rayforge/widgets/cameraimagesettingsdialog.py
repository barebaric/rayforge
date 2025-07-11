import logging
from gi.repository import Gtk, Adw
from ..models.camera import Camera
from .cameradisplay import CameraDisplay


logger = logging.getLogger(__name__)


class CameraImageSettingsDialog(Adw.MessageDialog):
    def __init__(self, parent, camera: Camera, **kwargs):
        super().__init__(
            transient_for=parent,
            modal=True,
            heading=f"{camera.name} - Camera Image Settings",
            close_response="cancel",
            **kwargs
        )
        self.camera = camera

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.set_extra_child(main_box)

        # Camera Display
        self.camera_display = CameraDisplay(self.camera)
        main_box.append(self.camera_display)

        # Settings
        settings_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
        )
        main_box.append(settings_box)
        preferences_group = Adw.PreferencesGroup(title="Camera Image Settings")
        settings_box.append(preferences_group)

        # White balance
        self.white_balance_adjustment = Gtk.Adjustment(
            value=self.camera.white_balance,
            lower=2500,
            upper=10000,
            step_increment=10,
            page_increment=100
        )
        self.white_balance_scale = Gtk.Scale.new(
            Gtk.Orientation.HORIZONTAL, self.white_balance_adjustment
        )
        self.white_balance_scale.set_hexpand(True)
        self.white_balance_scale.set_digits(0)  # White balance in Kelvin
        self.white_balance_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self.white_balance_scale.connect(
            "value-changed", self.on_white_balance_changed
        )
        white_balance_row = Adw.ActionRow(title="White Balance")
        white_balance_row.add_suffix(self.white_balance_scale)
        preferences_group.add(white_balance_row)

        # Contrast
        self.contrast_adjustment = Gtk.Adjustment(
            value=self.camera.contrast,
            lower=0.0,
            upper=100.0,
            step_increment=0.01,
            page_increment=10.0
        )
        self.contrast_scale = Gtk.Scale.new(
            Gtk.Orientation.HORIZONTAL, self.contrast_adjustment
        )
        self.contrast_scale.set_hexpand(True)
        self.contrast_scale.set_digits(2)  # Contrast can have two decimals
        self.contrast_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self.contrast_scale.connect("value-changed", self.on_contrast_changed)
        contrast_row = Adw.ActionRow(title="Contrast")
        contrast_row.add_suffix(self.contrast_scale)
        preferences_group.add(contrast_row)

        # Brightness
        self.brightness_adjustment = Gtk.Adjustment(
            value=self.camera.brightness,
            lower=-100.0,
            upper=100.0,
            step_increment=0.01,
            page_increment=10.0
        )
        self.brightness_scale = Gtk.Scale.new(
            Gtk.Orientation.HORIZONTAL, self.brightness_adjustment
        )
        self.brightness_scale.set_hexpand(True)
        self.brightness_scale.set_digits(2)
        self.brightness_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self.brightness_scale.connect(
            "value-changed", self.on_brightness_changed
        )
        brightness_row = Adw.ActionRow(title="Brightness")
        brightness_row.add_suffix(self.brightness_scale)
        preferences_group.add(brightness_row)

        # Add buttons
        self.add_response("close", "Close")
        self.set_default_response("cancel")
        self.connect("response", self.on_dialog_response)

    def on_white_balance_changed(self, adjustment):
        self.camera.white_balance = adjustment.get_value()

    def on_contrast_changed(self, adjustment):
        self.camera.contrast = adjustment.get_value()

    def on_brightness_changed(self, adjustment):
        self.camera.brightness = adjustment.get_value()

    def on_dialog_response(self, dialog, response_id):
        if response_id == "close" or response_id == "cancel":
            logger.debug(
                "CameraImageSettingsDialog closing, calling "
                f"CameraDisplay.stop() for camera {self.camera.name}"
            )
            self.camera_display.stop()
            self.close()
