"""CNC safe-Z row widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class SafeZRow(SpinRow):
    """A spin row bound to ``CncAssemblerStep.safe_z``."""

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(
            editor,
            step,
            "safe_z",
            _("Safe Z Height"),
            _("Height to retract between moves"),
            0.0,
            50.0,
            0.1,
            2,
            quantity="length",
        )
