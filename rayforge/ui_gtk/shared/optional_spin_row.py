from typing import Optional

from blinker import Signal
from gi.repository import Adw, Gtk

from ...context import get_context
from ...shared.units.definitions import get_unit


class OptionalSpinRowController:
    """Manages an ActionRow with a SpinButton and a Switch."""

    def __init__(
        self,
        group: Adw.PreferencesGroup,
        title: str,
        subtitle: str,
        quantity: str,
    ):
        self.changed = Signal()
        self.quantity = quantity

        config = get_context().config
        unit_name = config.unit_preferences.get(self.quantity)
        self.unit = get_unit(unit_name) if unit_name else None
        if not self.unit:
            raise ValueError(
                f"Could not determine unit for quantity '{quantity}'"
            )

        self.row = Adw.ActionRow(title=title, subtitle=subtitle)
        group.add(self.row)

        adj = Gtk.Adjustment(lower=0, upper=9999, step_increment=0.1)
        self.spin_button = Gtk.SpinButton(
            adjustment=adj, digits=self.unit.precision
        )
        self.spin_button.set_valign(Gtk.Align.CENTER)

        self.switch = Gtk.Switch(valign=Gtk.Align.CENTER)

        self.row.add_suffix(self.switch)
        self.row.add_suffix(self.spin_button)

        self.switch.connect("notify::active", self._on_toggled)
        self._value_changed_handler_id = self.spin_button.connect(
            "value-changed", lambda btn: self.changed.send(self)
        )

        self._on_toggled(self.switch, None)

    def _on_toggled(self, switch, _pspec):
        is_active = switch.get_active()
        self.spin_button.set_sensitive(is_active)
        self.changed.send(self)

    def get_value(self) -> Optional[float]:
        """Gets the value in base units, or None if disabled."""
        if not self.switch.get_active():
            return None
        return self.get_spin_value_in_base()

    def set_value(self, value_in_base: Optional[float]):
        """Sets the value from base units, or disables if None."""
        if value_in_base is None:
            self.switch.set_active(False)
            self.set_spin_value_in_base(0)
        else:
            self.switch.set_active(True)
            self.set_spin_value_in_base(value_in_base)

    def get_spin_value_in_base(self) -> float:
        """Gets the spinbutton's value in base units, ignoring the switch."""
        if not self.unit:
            return 0.0
        display_value = self.spin_button.get_value()
        return self.unit.to_base(display_value)

    def set_spin_value_in_base(self, value_in_base: float):
        """
        Sets the spinbutton's value from base units, without touching the
        switch.
        """
        if not self.unit:
            return
        self.spin_button.handler_block(self._value_changed_handler_id)
        display_value = self.unit.from_base(value_in_base)
        self.spin_button.set_value(display_value)
        self.spin_button.handler_unblock(self._value_changed_handler_id)
