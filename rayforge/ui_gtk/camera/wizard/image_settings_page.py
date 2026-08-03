"""Camera wizard page: image settings (resolution, WB, brightness, ...)."""

from gettext import gettext as _

from gi.repository import Gtk

from ..image_settings_widget import CameraImageSettings
from .base_page import CameraWizardPage


class ImageSettingsPage(CameraWizardPage):
    step_name = "image"
    title = _("Image Settings")

    def __init__(self, wizard, controller):
        super().__init__(wizard, controller)
        self._widget: CameraImageSettings | None = None

    def build(self) -> Gtk.Box:
        self.root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._widget = CameraImageSettings(self.controller)
        self.root.append(self._widget)
        return self.root

    def leave(self) -> None:
        if self._widget is not None:
            self._widget.stop()


__all__ = ["ImageSettingsPage"]
