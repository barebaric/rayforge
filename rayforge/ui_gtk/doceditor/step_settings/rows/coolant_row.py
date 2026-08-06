"""Coolant method selection row for step settings."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw
from raygeo.ops.state import CoolantMode

from rayforge.ui_gtk.icons import get_icon

from .combo_row import ComboRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class CoolantRow(ComboRow):
    """A combo row bound to the ``coolant_method`` step attribute.

    Shows an exclamation mark when the selected method is not
    supported by the machine's currently selected head.
    """

    def __init__(self, editor: "DocEditor", step: Any):
        choices = [
            (_("Off"), CoolantMode.OFF),
            (_("Flood"), CoolantMode.FLOOD),
            (_("Mist"), CoolantMode.MIST),
        ]
        super().__init__(
            editor,
            step,
            "coolant_method",
            _("Cooling"),
            choices,
            _("Coolant delivered to the workpiece while cutting"),
        )

    def build_widget(self) -> Adw.ComboRow:
        row = super().build_widget()
        self._warning_icon = get_icon("warning-symbolic")
        self._warning_icon.set_tooltip_text(
            _("This cooling method is not supported by the current machine")
        )
        self._warning_icon.set_visible(False)
        row.add_suffix(self._warning_icon)
        return row

    def _sync_dependencies(self):
        machine = self.get_machine()
        if machine is None:
            self._warning_icon.set_visible(False)
            return
        unsupported = self.step.get_unsupported_coolant_methods(machine)
        self._warning_icon.set_visible(bool(unsupported))
