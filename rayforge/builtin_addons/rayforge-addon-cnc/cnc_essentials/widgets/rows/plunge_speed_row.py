"""CNC plunge speed row widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class PlungeSpeedRow(SpinRow):
    """A spin row bound to ``CncAssemblerStep.plunge_speed``."""

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(
            editor,
            step,
            "plunge_speed",
            _("Plunge Rate"),
            _("Vertical feed rate"),
            1.0,
            float(step.max_cut_speed),
            10.0,
            0,
            is_int=True,
            quantity="speed",
        )

    def _sync_dependencies(self):
        if self.step.max_cut_speed:
            self.set_range(1.0, float(self.step.max_cut_speed))
