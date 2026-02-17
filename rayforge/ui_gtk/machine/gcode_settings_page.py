import logging
from gi.repository import Gtk, Adw
from ..shared.adwfix import get_spinrow_int
from .dialect_list import DialectListEditor


logger = logging.getLogger(__name__)


class GcodeSettingsPage(Adw.PreferencesPage):
    def __init__(self, machine, **kwargs):
        super().__init__(
            title=_("G-code"),
            icon_name="gcode-symbolic",
            **kwargs,
        )
        self.machine = machine

        precision_group = Adw.PreferencesGroup(title=_("Precision"))
        precision_group.set_description(
            _("Configure the numeric precision of coordinate output.")
        )
        self.add(precision_group)

        precision_adjustment = Gtk.Adjustment(
            lower=1, upper=8, step_increment=1, page_increment=1
        )
        self.precision_row = Adw.SpinRow(
            title=_("G-code Precision"),
            subtitle=_("Number of decimal places for coordinates"),
            adjustment=precision_adjustment,
        )
        precision_adjustment.set_value(self.machine.gcode_precision)
        self.precision_row.connect("changed", self.on_precision_changed)
        precision_group.add(self.precision_row)

        dialect_editor_group = DialectListEditor(
            machine=self.machine,
            title=_("Dialect"),
            description=_(
                "Select, create and manage G-code dialect definitions."
            ),
        )
        self.add(dialect_editor_group)

    def on_precision_changed(self, spinrow):
        """Update the machine's G-code precision when the value changes."""
        value = get_spinrow_int(spinrow)
        self.machine.set_gcode_precision(value)
