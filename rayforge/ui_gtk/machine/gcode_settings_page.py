import logging
from gettext import gettext as _

from gi.repository import Adw

from ..shared.preferences_page import TrackedPreferencesPage
from ..shared.unit_spin_row import SpinRow
from .dialect_list import DialectListEditor

logger = logging.getLogger(__name__)


class GcodeSettingsPage(TrackedPreferencesPage):
    key = "gcode"
    path_prefix = "/machine-settings/"

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

        self.precision_row = SpinRow(
            _("G-code Precision"),
            _("Number of decimal places for coordinates"),
            lower=1,
            upper=8,
            page_increment=1,
            value=self.machine.gcode_precision,
        )
        self.precision_row.value_changed.connect(self.on_precision_changed)
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
        value = spinrow.get_int_value()
        self.machine.set_gcode_precision(value)
