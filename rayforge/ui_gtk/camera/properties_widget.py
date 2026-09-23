import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...camera.controller import CameraController
from ...camera.models.camera import Camera, CameraSourceType
from ...camera.source import validate_source_uri
from ...camera.v4l import display_name
from ..icons import get_icon
from .alignment_dialog import CameraAlignmentDialog
from .image_settings_dialog import CameraImageSettingsDialog
from .lens_calibration_dialog import LensCalibrationDialog

logger = logging.getLogger(__name__)


class CameraProperties(Adw.PreferencesGroup):
    def __init__(self, controller: CameraController | None, **kwargs):
        super().__init__(**kwargs)
        self._controller: CameraController | None = None
        self._camera: Camera | None = None
        self._updating_ui: bool = False
        # True while we are writing a device_id chosen by the user in
        # the local-device dropdown. While true, the resulting
        # `changed` signal must not trigger a fresh hardware scan: the
        # capture stream is asynchronously opening the newly selected
        # device around this same time, and probing it again
        # concurrently (via cv2.VideoCapture) can crash the V4L2/
        # DirectShow driver.
        self._committing_local_source: bool = False
        # Identifies the source the local-device dropdown was last
        # populated for. A `changed` signal that leaves this key
        # untouched (e.g. a distortion coefficient edit) must not
        # trigger another hardware scan: probing every V4L2 device is
        # slow and blocks the UI thread, and doing it repeatedly can
        # collide with a capture stream reopening the same device.
        self._last_scanned_source_key: tuple | None = None

        self.set_title(_("Camera Properties"))
        self.set_description(_("Configure the selected camera."))

        self.source_row = Adw.ComboRow(
            title=_("Source"),
            subtitle=_("Choose a connected local camera"),
            model=Gtk.StringList.new([]),
        )
        self.source_row.connect(
            "notify::selected", self._on_local_source_selected
        )
        self.add(self.source_row)

        self.source_entry_row = Adw.ActionRow(
            title=_("Source"),
            subtitle=_("Camera source URL"),
        )
        self.source_entry = Gtk.Entry()
        self.source_entry.set_valign(Gtk.Align.CENTER)
        self.source_entry.set_width_chars(40)
        self.source_entry.set_hexpand(True)
        self.source_entry.connect("changed", self._on_source_changed)
        self.source_entry.connect("activate", self._commit_source_uri)
        focus_controller = Gtk.EventControllerFocus()
        focus_controller.connect("leave", self._commit_source_uri)
        self.source_entry.add_controller(focus_controller)
        self.source_entry_row.add_suffix(self.source_entry)
        self.add(self.source_entry_row)
        self.source_combo = self.source_row
        self._local_device_ids: list[str] = []

        # Camera Name
        self.name_row = Adw.ActionRow(
            title=_("Name"),
            subtitle=_("Display name for this camera"),
        )
        self.name_entry = Gtk.Entry()
        self.name_entry.set_valign(Gtk.Align.CENTER)
        self.name_entry.connect("changed", self.on_name_changed)
        self.name_row.add_suffix(self.name_entry)
        self.add(self.name_row)

        # Enabled Switch
        self.enabled_row = Adw.ActionRow(
            title=_("Enabled"),
            subtitle=_("Turn the camera stream on or off"),
        )
        self.enabled_switch = Gtk.Switch()
        self.enabled_switch.set_valign(Gtk.Align.CENTER)
        self.enabled_switch.connect("notify::active", self.on_enabled_changed)
        self.enabled_row.add_suffix(self.enabled_switch)
        self.enabled_row.set_activatable_widget(self.enabled_switch)
        self.add(self.enabled_row)

        # Camera Wizard — runs the full guided setup (image settings,
        # lens calibration, alignment) in one flow.
        self.wizard_button = Gtk.Button(
            label=_("Start"), valign=Gtk.Align.CENTER
        )
        self.wizard_button.add_css_class("suggested-action")
        self.wizard_button.connect("clicked", self.on_wizard_button_clicked)
        wizard_row = Adw.ActionRow(
            title=_("Camera Wizard"),
            subtitle=_(
                "Guided setup: image settings, lens calibration, "
                "and alignment."
            ),
        )
        wizard_row.add_suffix(self.wizard_button)
        wizard_row.set_activatable_widget(self.wizard_button)
        self.add(wizard_row)

        # Image Settings button
        self.image_settings_button = Gtk.Button(
            label=_("Configure"), valign=Gtk.Align.CENTER
        )
        self.image_settings_button.connect(
            "clicked", self.on_image_settings_button_clicked
        )
        image_settings_row = Adw.ActionRow(
            title=_("Image Settings"),
            subtitle=_(
                "Adjust brightness, contrast, white balance, and noise"
            ),
        )
        image_settings_row.add_suffix(self.image_settings_button)
        self.add(image_settings_row)

        # Lens Calibration
        self.lens_calibration_button = Gtk.Button(
            label=_("Configure"),
            valign=Gtk.Align.CENTER,
            margin_start=6,
        )
        self.lens_calibration_button.connect(
            "clicked", self.on_lens_calibration_button_clicked
        )
        self.lens_calibration_row = Adw.ActionRow(
            title=_("Lens Calibration"),
            subtitle=_("Correct lens distortion for straighter lines"),
        )
        self._cal_ok = get_icon("check-circle-symbolic")
        self._cal_ok.set_valign(Gtk.Align.CENTER)
        self._cal_ok.set_visible(False)
        self._cal_warn = get_icon("warning-symbolic")
        self._cal_warn.set_valign(Gtk.Align.CENTER)
        self._cal_warn.set_visible(False)
        self.lens_calibration_row.add_suffix(self._cal_ok)
        self.lens_calibration_row.add_suffix(self._cal_warn)
        self.lens_calibration_row.add_suffix(self.lens_calibration_button)
        self.add(self.lens_calibration_row)

        # Image Alignment
        self.image_alignment_button = Gtk.Button(
            label=_("Configure"),
            valign=Gtk.Align.CENTER,
            margin_start=6,
        )
        self.image_alignment_button.connect(
            "clicked", self.on_image_alignment_button_clicked
        )
        self.image_alignment_row = Adw.ActionRow(
            title=_("Image Alignment"),
            subtitle=_("Calibrate camera position and perspective"),
        )
        self._align_ok = get_icon("check-circle-symbolic")
        self._align_ok.set_valign(Gtk.Align.CENTER)
        self._align_ok.set_visible(False)
        self._align_warn = get_icon("warning-symbolic")
        self._align_warn.set_valign(Gtk.Align.CENTER)
        self._align_warn.set_visible(False)
        self.image_alignment_row.add_suffix(self._align_ok)
        self.image_alignment_row.add_suffix(self._align_warn)
        self.image_alignment_row.add_suffix(self.image_alignment_button)
        self.add(self.image_alignment_row)

        self.set_controller(controller)

    def set_controller(self, controller: CameraController | None):
        if self._camera:
            self._camera.changed.disconnect(self._on_camera_changed)

        self._controller = controller
        self._camera = controller.config if controller else None
        self._last_scanned_source_key = None

        if self._camera:
            self._camera.changed.connect(self._on_camera_changed)
            self.update_ui()
            self.set_sensitive(True)
        else:
            self.clear_ui()
            self.set_sensitive(False)

    def update_ui(self, rescan_local_devices: bool = True):
        if not self._camera:
            self.clear_ui()
            return
        if self._updating_ui:
            return

        self._updating_ui = True
        try:
            if self._camera.source_type is CameraSourceType.LOCAL_DEVICE:
                self.source_combo.set_subtitle(
                    _("Choose a connected local camera")
                )
                if rescan_local_devices:
                    self._update_local_device_choices()
                self.source_entry.set_visible(False)
                self.source_entry_row.set_visible(False)
                self.source_combo.set_visible(True)
            else:
                self.source_entry_row.set_subtitle(_("Camera source URL"))
                self.source_entry.set_visible(True)
                self.source_entry_row.set_visible(True)
                self.source_combo.set_visible(False)
                self.source_entry.set_text(self._camera.source_uri)
                self._validate_source_uri()
            self.name_entry.set_text(self._camera.name)
            self.enabled_switch.set_active(self._camera.enabled)
            self.image_settings_button.set_sensitive(self._camera.enabled)
            self.lens_calibration_button.set_sensitive(self._camera.enabled)
            self.image_alignment_button.set_sensitive(self._camera.enabled)
            self.wizard_button.set_sensitive(self._camera.enabled)
            self._update_status_icons()
        finally:
            self._updating_ui = False

    def _update_local_device_choices(self) -> None:
        if not self._camera:
            return
        device_ids = CameraController.list_available_devices()
        self._last_scanned_source_key = self._source_key()
        current_id = self._camera.device_id
        current_missing = bool(current_id and current_id not in device_ids)
        if current_missing:
            device_ids.append(current_id)
        current_active = bool(
            current_missing
            and self._controller
            and self._controller.has_active_source
        )

        self._local_device_ids = device_ids
        labels = [
            (
                _("Unavailable: {device_id}").format(device_id=device_id)
                if (
                    current_missing
                    and not current_active
                    and device_id == current_id
                )
                else display_name(device_id)
            )
            for device_id in device_ids
        ]
        self.source_combo.set_model(Gtk.StringList.new(labels))
        if current_id in device_ids:
            self.source_combo.set_selected(device_ids.index(current_id))

    def _on_local_source_selected(self, combo: Gtk.DropDown, _pspec) -> None:
        if (
            self._updating_ui
            or not self._camera
            or self._camera.source_type is not CameraSourceType.LOCAL_DEVICE
        ):
            return
        index = combo.get_selected()
        if 0 <= index < len(self._local_device_ids):
            device_id = self._local_device_ids[index]
            if device_id != self._camera.device_id:
                self._committing_local_source = True
                try:
                    self._camera.device_id = device_id
                finally:
                    self._committing_local_source = False

    def _update_status_icons(self):
        cam = self._camera
        if not cam:
            return

        calibrated = cam.calibration_date is not None
        self._cal_ok.set_visible(calibrated)
        self._cal_warn.set_visible(not calibrated)
        if calibrated:
            self._cal_ok.set_tooltip_text(_("Lens calibration completed"))
        else:
            self._cal_warn.set_tooltip_text(
                _("Lens calibration not yet performed")
            )

        valid = cam.alignment_valid
        stale = cam.has_alignment and not valid
        self._align_ok.set_visible(valid)
        self._align_warn.set_visible(not valid)
        if valid:
            self._align_ok.set_tooltip_text(_("Image alignment completed"))
        elif stale:
            self._align_warn.set_tooltip_text(
                _(
                    "Image alignment must be redone after lens "
                    "calibration was updated"
                )
            )
        else:
            self._align_warn.set_tooltip_text(
                _("Image alignment not yet performed")
            )

    def clear_ui(self):
        self.source_combo.set_subtitle("")
        self.source_entry.set_text("")
        self.source_entry.set_visible(False)
        self.source_entry_row.set_visible(False)
        self.source_entry.remove_css_class("error")
        self.source_entry.set_tooltip_text(None)
        self.source_combo.set_model(Gtk.StringList.new([]))
        self.source_combo.set_visible(False)
        self._local_device_ids = []
        self._last_scanned_source_key = None
        self.name_entry.set_text("")
        self.enabled_switch.set_active(False)
        # Clear image settings and disable button
        self.image_settings_button.set_sensitive(False)
        self.lens_calibration_button.set_sensitive(False)
        self.image_alignment_button.set_sensitive(False)
        self.wizard_button.set_sensitive(False)

    def _source_key(self) -> tuple | None:
        """Identifies which source the device dropdown must reflect."""
        if not self._camera:
            return None
        return (self._camera.source_type, self._camera.device_id)

    def _on_camera_changed(self, camera, *args):
        logger.debug("Camera model changed, updating UI for %s", camera.name)
        # Only re-probe the hardware when the change could actually
        # affect the device list. Unrelated edits (distortion
        # coefficients, name, enabled state, ...) must not trigger a
        # blocking scan of every V4L2 device.
        rescan = (
            not self._committing_local_source
            and self._source_key() != self._last_scanned_source_key
        )
        self.update_ui(rescan_local_devices=rescan)

    def _validate_source_uri(self) -> bool:
        if not self._camera or self._camera.source_type is (
            CameraSourceType.LOCAL_DEVICE
        ):
            return True
        error = validate_source_uri(
            self._camera.source_type, self.source_entry.get_text()
        )
        self.source_entry.set_css_classes(["error"] if error else [])
        self.source_entry.set_tooltip_text(error)
        return error is None

    def _on_source_changed(self, entry: Gtk.Entry) -> None:
        if self._updating_ui:
            return
        self._validate_source_uri()

    def _commit_source_uri(self, *args) -> None:
        if (
            not self._camera
            or self._camera.source_type is CameraSourceType.LOCAL_DEVICE
            or self._updating_ui
            or not self._validate_source_uri()
        ):
            return
        uri = self.source_entry.get_text().strip()
        if uri != self._camera.source_uri:
            self._camera.source_uri = uri

    def on_name_changed(self, entry_row):
        if not self._camera or self._updating_ui:
            return
        self._updating_ui = True
        try:
            self._camera.name = entry_row.get_text()
        finally:
            self._updating_ui = False

    def on_enabled_changed(self, switch_row, _):
        if not self._camera:
            return
        self._camera.enabled = switch_row.get_active()

    def on_image_settings_button_clicked(self, button):
        """Open the CameraImageSettingsDialog."""
        if not self._controller:
            return
        window = self.get_ancestor(Gtk.Window)
        if isinstance(window, Gtk.Window):
            dialog = CameraImageSettingsDialog(window, self._controller)
            dialog.present()

    def on_wizard_button_clicked(self, button):
        """Launch the guided camera wizard."""
        if not self._controller:
            return
        window = self.get_ancestor(Gtk.Window)
        if not isinstance(window, Gtk.Window):
            return
        from .wizard.wizard import CameraWizard

        wizard = CameraWizard(window, self._controller)
        wizard.present()

    def on_lens_calibration_button_clicked(self, button):
        if not self._controller:
            return
        window = self.get_ancestor(Gtk.Window)
        if isinstance(window, Gtk.Window):
            dialog = LensCalibrationDialog(window, self._controller)
            dialog.present()

    def on_image_alignment_button_clicked(self, button):
        """Open the CameraImageAlignmentDialog."""
        if not self._controller:
            return
        window = self.get_ancestor(Gtk.Window)
        if isinstance(window, Gtk.Window):
            dialog = CameraAlignmentDialog(window, self._controller)
            dialog.present()
