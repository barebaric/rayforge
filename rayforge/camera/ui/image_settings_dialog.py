import logging
from gi.repository import Gtk, Adw  # type: ignore
from ..models.camera import Camera
from .display_widget import CameraDisplay


logger = logging.getLogger(__name__)


class CameraImageSettingsDialog(Adw.MessageDialog):
    def __init__(self, parent, camera: Camera, **kwargs):
        super().__init__(
            transient_for=parent,
            modal=True,
            heading=_("{camera_name} - Camera Image Settings").format(
                camera_name=camera.name
            ),
            close_response="cancel",
            **kwargs,
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
        preferences_group = Adw.PreferencesGroup(
            title=_("Camera Image Settings")
        )
        settings_box.append(preferences_group)

        # White balance
        self.auto_white_balance_switch = Adw.SwitchRow(
            title=_("Auto White Balance"),
            subtitle=_("Automatically adjust white balance"),
        )
        self.auto_white_balance_switch.set_active(
            self.camera.white_balance is None
        )
        self.auto_white_balance_switch.connect(
            "notify::active", self.on_auto_white_balance_toggled
        )
        preferences_group.add(self.auto_white_balance_switch)

        self.wb_adjustment = Gtk.Adjustment(
            lower=2500,
            upper=10000,
            step_increment=10,
            page_increment=100,
        )
        self.white_balance_scale = Gtk.Scale.new(
            Gtk.Orientation.HORIZONTAL, self.wb_adjustment
        )
        self.wb_adjustment.set_value(
            self.camera.white_balance
            if self.camera.white_balance is not None
            else 4000
        )
        self.white_balance_scale.set_size_request(300, -1)
        self.white_balance_scale.set_digits(0)  # White balance in Kelvin
        self.white_balance_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self.white_balance_scale.connect(
            "value-changed", self.on_white_balance_changed
        )
        white_balance_row = Adw.ActionRow(title=_("White Balance (Kelvin)"))
        white_balance_row.add_suffix(self.white_balance_scale)
        preferences_group.add(white_balance_row)

        # Set initial sensitivity based on current camera setting
        self.white_balance_scale.set_sensitive(
            self.camera.white_balance is not None
        )

        # Contrast
        self.contrast_adjustment = Gtk.Adjustment(
            lower=0.0,
            upper=100.0,
            step_increment=0.01,
            page_increment=10.0,
        )
        self.contrast_scale = Gtk.Scale.new(
            Gtk.Orientation.HORIZONTAL, self.contrast_adjustment
        )
        self.contrast_adjustment.set_value(self.camera.contrast)
        self.contrast_scale.set_size_request(300, -1)
        self.contrast_scale.set_digits(2)  # Contrast can have two decimals
        self.contrast_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self.contrast_scale.connect("value-changed", self.on_contrast_changed)
        contrast_row = Adw.ActionRow(title=_("Contrast"))
        contrast_row.add_suffix(self.contrast_scale)
        preferences_group.add(contrast_row)

        # Brightness
        self.brightness_adjustment = Gtk.Adjustment(
            lower=-100.0,
            upper=100.0,
            step_increment=0.01,
            page_increment=10.0,
        )
        self.brightness_scale = Gtk.Scale.new(
            Gtk.Orientation.HORIZONTAL, self.brightness_adjustment
        )
        self.brightness_adjustment.set_value(self.camera.brightness)
        self.brightness_scale.set_size_request(300, -1)
        self.brightness_scale.set_digits(2)
        self.brightness_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self.brightness_scale.connect(
            "value-changed", self.on_brightness_changed
        )
        brightness_row = Adw.ActionRow(title=_("Brightness"))
        brightness_row.add_suffix(self.brightness_scale)
        preferences_group.add(brightness_row)

        # Transparency
        self.transparency_adjustment = Gtk.Adjustment(
            lower=0.0,
            upper=1.0,
            step_increment=0.01,
            page_increment=0.1,
        )
        self.transparency_scale = Gtk.Scale.new(
            Gtk.Orientation.HORIZONTAL, self.transparency_adjustment
        )
        self.transparency_adjustment.set_value(self.camera.transparency)
        self.transparency_scale.set_size_request(300, -1)
        self.transparency_scale.set_digits(2)
        self.transparency_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self.transparency_scale.connect(
            "value-changed", self.on_transparency_changed
        )
        transparency_row = Adw.ActionRow(
            title=_("Transparency"),
            subtitle=_("Transparency on the worksurface"),
        )
        transparency_row.add_suffix(self.transparency_scale)
        preferences_group.add(transparency_row)

        # Add buttons
        self.add_response("close", _("Close"))
        self.set_default_response("cancel")
        self.connect("response", self.on_dialog_response)

    def on_white_balance_changed(self, adjustment):
        if not self.auto_white_balance_switch.get_active():
            self.camera.white_balance = adjustment.get_value()

    def on_auto_white_balance_toggled(self, switch_row, pspec):
        is_auto = switch_row.get_active()
        self.white_balance_scale.set_sensitive(not is_auto)
        if is_auto:
            self.camera.white_balance = None
        else:
            self.camera.white_balance = self.wb_adjustment.get_value()

    def on_contrast_changed(self, adjustment):
        self.camera.contrast = adjustment.get_value()

    def on_brightness_changed(self, adjustment):
        self.camera.brightness = adjustment.get_value()

    def on_transparency_changed(self, adjustment):
        self.camera.transparency = adjustment.get_value()

    def on_dialog_response(self, dialog, response_id):
        if response_id == "close" or response_id == "cancel":
            logger.debug(
                "CameraImageSettingsDialog closing, calling "
                f"CameraDisplay.stop() for camera {self.camera.name}"
            )
            self.camera_display.stop()
            self.close()
