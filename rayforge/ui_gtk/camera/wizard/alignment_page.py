"""Camera wizard page: image↔world alignment."""

from gettext import gettext as _
from typing import List

from blinker import Signal
from gi.repository import Gtk

from ....camera.controller import CameraController
from ..alignment_widget import CameraAlignment
from .base_page import CameraWizardPage


class AlignmentPage(CameraWizardPage):
    step_name = "alignment"
    title = _("Image Alignment")

    def __init__(self, wizard, controller: CameraController):
        super().__init__(wizard, controller)
        # Fired when the user applies the alignment.
        self.alignment_applied = Signal()
        self._widget: CameraAlignment | None = None

    def build(self) -> Gtk.Box:
        self.root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._widget = CameraAlignment(self.controller)
        self._widget.applied.connect(self._on_applied)
        self.root.append(self._widget)
        return self.root

    def leave(self) -> None:
        if self._widget is not None:
            self._widget.stop()

    def can_proceed(self) -> bool:
        return False

    def footer_buttons(self) -> List[Gtk.Button]:
        if self._widget is not None:
            return self._widget.footer_buttons()
        return []

    def _on_applied(self, _sender) -> None:
        self.alignment_applied.send(self)


__all__ = ["AlignmentPage"]
