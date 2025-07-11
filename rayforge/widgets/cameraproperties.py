from gi.repository import Gtk, Adw
from typing import Optional
import logging
from ..models.camera import Camera
from ..util.adwfix import get_spinrow_float


logger = logging.getLogger(__name__)


class CameraProperties(Adw.PreferencesGroup):
    def __init__(self, camera: Optional[Camera], **kwargs):
        super().__init__(**kwargs)
        self._camera: Optional[Camera] = None
        self._updating_ui: bool = False

        self.set_title("Camera Properties")
        self.set_description("Configure the selected camera")

        # Device ID
        self.device_id_row = Adw.ActionRow(title="Device ID")
        self.add(self.device_id_row)

        # Camera Name
        self.name_row = Adw.EntryRow(title="Name")
        self.name_row.connect("changed", self.on_name_changed)
        self.add(self.name_row)

        # Enabled Switch
        self.enabled_row = Adw.SwitchRow(title="Enabled")
        self.enabled_row.connect("notify::active", self.on_enabled_changed)
        self.add(self.enabled_row)

        # Image Dimensions
        self.width_mm_row = Adw.SpinRow(
            title="Width (mm)",
            subtitle="Width of the camera image in mm",
            adjustment=Gtk.Adjustment(
                value=0, lower=0, upper=1000, step_increment=0.1,
                page_increment=1
            ),
            digits=3
        )
        self.width_mm_row.connect("changed", self.on_width_mm_changed)
        self.add(self.width_mm_row)

        self.height_mm_row = Adw.SpinRow(
            title="Height (mm)",
            subtitle="Height of the camera image in mm",
            adjustment=Gtk.Adjustment(
                value=0, lower=0, upper=1000, step_increment=0.1,
                page_increment=1
            ),
            digits=3
        )
        self.height_mm_row.connect("changed", self.on_height_mm_changed)
        self.add(self.height_mm_row)

        self.set_camera(camera)

    def set_camera(self, camera: Optional[Camera]):
        if self._camera:
            self._camera.changed.disconnect(self._on_camera_changed)
        self._camera = camera
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
            self.name_row.set_text(self._camera.name)
            self.enabled_row.set_active(self._camera.enabled)
            self.width_mm_row.set_value(self._camera.width_mm)
            self.height_mm_row.set_value(self._camera.height_mm)
        finally:
            self._updating_ui = False

    def clear_ui(self):
        self.device_id_row.set_subtitle("")
        self.name_row.set_text("")
        self.enabled_row.set_active(False)
        self.width_mm_row.set_value(0.0)
        self.height_mm_row.set_value(0.0)

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
        if not self._camera or self._updating_ui:
            return
        self._updating_ui = True
        try:
            self._camera.enabled = switch_row.get_active()
        finally:
            self._updating_ui = False

    def on_width_mm_changed(self, spin_row):
        if not self._camera or self._updating_ui:
            return
        self._updating_ui = True
        try:
            self._camera.width_mm = get_spinrow_float(spin_row)
        finally:
            self._updating_ui = False

    def on_height_mm_changed(self, spin_row):
        if not self._camera or self._updating_ui:
            return
        self._updating_ui = True
        try:
            self._camera.height_mm = get_spinrow_float(spin_row)
        finally:
            self._updating_ui = False
