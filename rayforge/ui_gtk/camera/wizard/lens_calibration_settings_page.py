"""Camera wizard page: manual lens-distortion coefficients."""

from gettext import gettext as _

from gi.repository import Gtk

from ....camera.controller import CameraController
from ..lens_calibration_widget import LensCalibrationWidget
from .base_page import CameraWizardPage


class LensCalibrationSettingsPage(CameraWizardPage):
    step_name = "lens_manual"
    title = _("Lens Calibration")

    def __init__(self, wizard, controller: CameraController):
        super().__init__(wizard, controller)
        self.camera = controller.config
        self._widget: LensCalibrationWidget | None = None

    def build(self) -> Gtk.Box:
        self.root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        right_scroll = Gtk.ScrolledWindow()
        right_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.root.append(right_scroll)
        right_scroll.set_vexpand(True)

        settings_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=12
        )
        settings_box.set_margin_start(12)
        settings_box.set_margin_end(12)
        settings_box.set_margin_top(12)
        settings_box.set_margin_bottom(12)
        right_scroll.set_child(settings_box)

        self._widget = LensCalibrationWidget(self.camera)
        settings_box.append(self._widget)
        return self.root

    def leave(self) -> None:
        if self._widget is not None:
            self._widget.stop()


__all__ = ["LensCalibrationSettingsPage"]
