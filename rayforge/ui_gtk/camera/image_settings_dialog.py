import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...camera.controller import CameraController
from ..shared.patched_dialog_window import PatchedDialogWindow
from .image_settings_widget import CameraImageSettings

logger = logging.getLogger(__name__)


class CameraImageSettingsDialog(PatchedDialogWindow):
    def __init__(self, parent, controller: CameraController, **kwargs):
        super().__init__(
            transient_for=parent,
            modal=True,
            default_width=1150,
            default_height=750,
            title=_("{camera_name} - Camera Image Settings").format(
                camera_name=controller.config.name
            ),
            **kwargs,
        )
        self.controller = controller

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(content)

        header = Adw.HeaderBar()
        content.append(header)

        self._widget = CameraImageSettings(controller)
        self._widget.set_margin_start(32)
        self._widget.set_margin_end(32)
        self._widget.set_margin_top(12)
        self._widget.set_margin_bottom(12)
        content.append(self._widget)

    def do_close_request(self, *args) -> bool:
        logger.debug(
            f"CameraImageSettingsDialog closing for "
            f"{self.controller.config.name}"
        )
        self._widget.stop()
        return False


__all__ = ["CameraImageSettingsDialog"]
