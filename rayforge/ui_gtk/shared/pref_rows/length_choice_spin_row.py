"""A length spin row with an inline unit-chooser dropdown."""

from gi.repository import Gtk

from ....shared.units.definitions import get_units_for_quantity
from .length_spin_row import LengthSpinRow


class LengthChoiceSpinRow(LengthSpinRow):
    """
    A length spin row with a unit chooser dropdown.

    Like :class:`LengthSpinRow`, values are exchanged with the caller in
    base units (mm), but instead of following the global display-unit
    preference the user picks the unit per row via a dropdown right of
    the spin button. It defaults to the configured preferred unit for
    ``length`` (e.g. ``mm``); switching the dropdown does not change the
    global preference and survives later preference changes.
    """

    __gtype_name__ = "RayforgeLengthChoiceSpinRow"

    def __init__(
        self,
        title: str,
        subtitle: str | None = None,
        **kwargs,
    ):
        self._unit_override: str | None = None
        self._units = get_units_for_quantity("length")
        self._dropdown_populated = False
        self._unit_dropdown = Gtk.DropDown()
        super().__init__(title, subtitle, **kwargs)

        self._populate_dropdown()
        self._sync_dropdown_to_unit()
        self._unit_dropdown.connect("notify::selected", self._on_unit_selected)
        self.add_suffix(self._unit_dropdown)

    def _resolve_unit_name(self) -> str | None:
        """Prefer the per-row choice over the global preference."""
        if self._unit_override is not None:
            return self._unit_override
        return super()._resolve_unit_name()

    def update_unit_and_bounds(self) -> None:
        super().update_unit_and_bounds()
        self._sync_dropdown_to_unit()

    def _populate_dropdown(self) -> None:
        string_list = Gtk.StringList()
        for unit in self._units:
            string_list.append(unit.label)
        self._unit_dropdown.set_model(string_list)
        self._unit_dropdown.set_valign(Gtk.Align.CENTER)
        self._dropdown_populated = True

    def _unit_index(self, unit_name: str) -> int:
        for i, unit in enumerate(self._units):
            if unit.name == unit_name:
                return i
        return 0

    def _sync_dropdown_to_unit(self) -> None:
        if self._unit is None or not self._dropdown_populated:
            return
        self._is_updating = True
        try:
            self._unit_dropdown.set_selected(self._unit_index(self._unit.name))
        finally:
            self._is_updating = False

    def _on_unit_selected(self, _dropdown, _pspec) -> None:
        if self._is_updating:
            return
        index = self._unit_dropdown.get_selected()
        if index < 0 or index >= len(self._units):
            return
        unit = self._units[index]
        if self._unit is not None and unit.name == self._unit.name:
            return
        # Preserve the semantic value while switching the display unit.
        base_value = self.get_value_in_base_units()
        self._unit_override = unit.name
        self._is_updating = True
        try:
            self.update_unit_and_bounds()
            if self._unit:
                self._spin_button.set_value(self._unit.from_base(base_value))
        finally:
            self._is_updating = False
