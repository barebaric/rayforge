from gi.repository import Gtk, Adw
from typing import Optional
import logging
from gettext import gettext as _
from ...camera.models.camera import Camera
from ...camera.controller import CameraController
from .image_settings_dialog import CameraImageSettingsDialog
from .alignment_dialog import CameraAlignmentDialog


logger = logging.getLogger(__name__)


class CameraProperties(Adw.PreferencesGroup):
    def __init__(self, controller: Optional[CameraController], **kwargs):
        super().__init__(**kwargs)
        self._controller: Optional[CameraController] = None
        self._camera: Optional[Camera] = None
        self._updating_ui: bool = False

        self.set_title(_("Camera Properties"))
        self.set_description(_("Configure the selected camera."))

        # Device ID
        self.device_id_row = Adw.ActionRow(
            title=_("Device ID"),
            subtitle=_("System identifier for the camera device"),
        )
        self.add(self.device_id_row)

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

        # Image alignment button
        self.image_alignment_button = Gtk.Button(
            label=_("Configure"), valign=Gtk.Align.CENTER
        )
        self.image_alignment_button.connect(
            "clicked", self.on_image_alignment_button_clicked
        )
        image_alignment_row = Adw.ActionRow(
            title=_("Image Alignment"),
            subtitle=_("Calibrate camera position and perspective"),
        )
        image_alignment_row.add_suffix(self.image_alignment_button)
        self.add(image_alignment_row)

        self.set_controller(controller)

    def set_controller(self, controller: Optional[CameraController]):
        if self._camera:
            self._camera.changed.disconnect(self._on_camera_changed)

        self._controller = controller
        self._camera = controller.config if controller else None

        if self._camera:
            self._camera.changed.connect(self._on_camera_changed)
            self.update_ui()
            self.set_sensitive(True)
        else:
            self.clear_ui()
            self.set_sensitive(False)

    def update_ui(self):
        if not self._camera:
            self.clear_ui()
            return
        if self._updating_ui:
            return

        self._updating_ui = True
        try:
            self.device_id_row.set_subtitle(self._camera.device_id)
            self.name_entry.set_text(self._camera.name)
            self.enabled_switch.set_active(self._camera.enabled)
            self.image_settings_button.set_sensitive(self._camera.enabled)
            self.image_alignment_button.set_sensitive(self._camera.enabled)
        finally:
            self._updating_ui = False

    def clear_ui(self):
        self.device_id_row.set_subtitle("")
        self.name_entry.set_text("")
        self.enabled_switch.set_active(False)
        # Clear image settings and disable button
        self.image_settings_button.set_sensitive(False)
        self.image_alignment_button.set_sensitive(False)

    def _on_camera_changed(self, camera, *args):
        logger.debug("Camera model changed, updating UI for %s", camera.name)
        self.update_ui()

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

    def on_image_alignment_button_clicked(self, button):
        """Open the CameraImageAlignmentDialog."""
        if not self._controller:
            return
        window = self.get_ancestor(Gtk.Window)
        if isinstance(window, Gtk.Window):
            dialog = CameraAlignmentDialog(window, self._controller)
            dialog.present()
