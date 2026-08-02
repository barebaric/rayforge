"""Laser air-assist row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SwitchRow


class AirAssistRow(SwitchRow):
    """A switch row bound to the ``LaserStep.air_assist`` attribute."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "air_assist",
            _("Air Assist"),
            _("Blow air over the cut to clear debris"),
        )
