"""Editor for the machine's speed-dependent bidirectional scan offset."""

from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...context import get_context
from ...shared.units.definitions import get_unit
from ..shared.pref_rows.length_spin_row import LengthSpinRow
from ..shared.pref_rows.speed_spin_row import SpeedSpinRow

#: Speed added above the fastest row when the user adds a row.
SPEED_STEP = 1000.0


def _format(quantity: str, value: float) -> str:
    unit_name = get_context().config.unit_preferences.get(quantity)
    unit = get_unit(unit_name) if unit_name else None
    if unit is None:
        return f"{value:g}"
    shown = unit.from_base(value)
    return f"{shown:.{unit.precision}f} {unit.label}"


class _OffsetEntryRow(Adw.ExpanderRow):
    """One ``(speed, offset)`` row of the table."""

    def __init__(self, speed: float, offset: float, on_change, on_remove):
        super().__init__()
        self.speed_row = SpeedSpinRow(
            _("Speed"),
            lower=1.0,
            upper=1_000_000.0,
            value_in_base=speed,
        )
        self.offset_row = LengthSpinRow(
            _("Offset"),
            _("Positive values shift right-to-left passes to the right"),
            lower=-5.0,
            upper=5.0,
            step_increment=0.001,
            digits=3,
            value_in_base=offset,
        )
        self.add_row(self.speed_row)
        self.add_row(self.offset_row)

        self.remove_button = Gtk.Button(icon_name="user-trash-symbolic")
        self.remove_button.set_valign(Gtk.Align.CENTER)
        self.remove_button.add_css_class("flat")
        self.remove_button.set_tooltip_text(_("Remove Row"))
        self.remove_button.connect("clicked", lambda _b: on_remove(self))
        self.add_suffix(self.remove_button)

        self.speed_row.value_changed.connect(lambda _r: on_change())
        self.offset_row.value_changed.connect(lambda _r: on_change())
        self.update_title()

    @property
    def entry(self) -> tuple[float, float]:
        return (
            self.speed_row.get_value_in_base_units(),
            self.offset_row.get_value_in_base_units(),
        )

    def update_title(self) -> None:
        speed, offset = self.entry
        self.set_title(
            _("{speed}: {offset}").format(
                speed=_format("speed", speed),
                offset=_format("length", offset),
            )
        )


class BidirOffsetTableGroup(Adw.PreferencesGroup):
    """Edits ``Machine.bidir_offset_table``."""

    def __init__(self, machine):
        super().__init__(title=_("Bidirectional Scan Offset by Speed"))
        self.set_description(
            _(
                "Shift of right-to-left raster passes at each engrave "
                "speed, interpolated between rows. Engrave steps use it "
                "when their own Bidirectional Scan Offset is 0."
            )
        )
        self.machine = machine
        self.entry_rows: list[_OffsetEntryRow] = []

        self.add_button = Gtk.Button(icon_name="list-add-symbolic")
        self.add_button.add_css_class("flat")
        self.add_button.set_tooltip_text(_("Add Row"))
        self.add_button.connect("clicked", self._on_add_clicked)
        self.set_header_suffix(self.add_button)

        self.empty_row = Adw.ActionRow(
            title=_("No rows: all speeds use an offset of 0")
        )
        self.empty_row.add_css_class("dim-label")
        self.add(self.empty_row)

        self._rebuild()

    def _rebuild(self) -> None:
        for row in self.entry_rows:
            self.remove(row)
        self.entry_rows = []
        for speed, offset in self.machine.bidir_offset_table:
            row = _OffsetEntryRow(
                speed, offset, self._on_row_changed, self._on_remove
            )
            self.entry_rows.append(row)
            self.add(row)
        self.empty_row.set_visible(not self.entry_rows)

    def _commit(self) -> None:
        self.machine.set_bidir_offset_table(
            [row.entry for row in self.entry_rows]
        )

    def _on_row_changed(self) -> None:
        for row in self.entry_rows:
            row.update_title()
        self._commit()

    def _on_add_clicked(self, _button) -> None:
        table = self.machine.bidir_offset_table
        if table:
            speed, offset = table[-1]
            entry = (speed + SPEED_STEP, offset)
        else:
            entry = (float(self.machine.max_cut_speed), 0.0)
        self.machine.set_bidir_offset_table([*table, entry])
        self._rebuild()

    def _on_remove(self, row: _OffsetEntryRow) -> None:
        self.remove(row)
        self.entry_rows.remove(row)
        self._commit()
        self._rebuild()
