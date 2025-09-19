import logging
from typing import Optional
from gi.repository import Adw
from blinker import Signal
from ...config import config
from ..units.definitions import Unit, get_unit
from ..util.adwfix import get_spinrow_int
from .formatter import format_value

logger = logging.getLogger(__name__)


class UnitSpinRowHelper:
    """
    A helper class that adds unit-aware functionality to a standard
    Adw.SpinRow.

    This class is not a widget. It is a controller that manages the state
    and logic for unit conversion, formatting, and bounds, and applies them
    directly to the provided SpinRow.
    """

    def __init__(
        self,
        spin_row: Adw.SpinRow,
        quantity: str,
        max_value_in_base: Optional[float] = None,
    ):
        self.spin_row = spin_row
        self.quantity = quantity
        self._unit: Unit | None = None
        self._is_updating = False
        self._original_subtitle_format = self.spin_row.get_subtitle() or ""
        self._max_value_in_base = max_value_in_base

        # Application-level signal for value changes (in base units)
        self.changed = Signal()

        # Connect to the adjustment's value-changed signal directly.
        adjustment = self.spin_row.get_adjustment()
        self._adj_handler_id = adjustment.connect(
            "value-changed", self._on_value_changed
        )
        self._config_handler_id = config.changed.connect(
            self._on_config_changed
        )
        self._destroy_handler_id = self.spin_row.connect(
            "destroy", self._on_destroy
        )

        self.update_format_and_bounds()

    def _on_destroy(self, _widget):
        adj = self.spin_row.get_adjustment()
        if adj and self._adj_handler_id:
            adj.disconnect(self._adj_handler_id)
        if self._config_handler_id:
            config.changed.disconnect(self._config_handler_id)
        self._adj_handler_id = None
        self._config_handler_id = None
        self._destroy_handler_id = None

    def _on_value_changed(self, adjustment):
        if not self._is_updating:
            self.changed.send(self)

    def _on_config_changed(self, sender, **kwargs):
        if not self._unit:
            self.update_format_and_bounds()
            return

        current_display_value = self.spin_row.get_value()
        base_value = self._unit.to_base(current_display_value)

        self._is_updating = True
        self.update_format_and_bounds()

        if self._unit:
            new_display_value = self._unit.from_base(base_value)
            self.spin_row.set_value(new_display_value)

        self._is_updating = False

    def update_format_and_bounds(self):
        """
        Sets the widget's unit, subtitle, and adjustment bounds based on
        config.
        """
        unit_name = config.unit_preferences.get(self.quantity)
        self._unit = get_unit(unit_name) if unit_name else None
        if not self._unit:
            return

        if self._max_value_in_base is not None:
            formatted_max = format_value(
                self._max_value_in_base, self.quantity
            )
            self.spin_row.set_subtitle(
                self._original_subtitle_format.format(max_speed=formatted_max)
            )
        else:
            self.spin_row.set_subtitle(
                f"{self._original_subtitle_format} ({self._unit.label})"
            )

        adj = self.spin_row.get_adjustment()
        if self._max_value_in_base is not None:
            new_upper = self._unit.from_base(self._max_value_in_base)
            adj.set_upper(new_upper)

        self.spin_row.set_digits(self._unit.precision)

    def set_value_in_base_units(self, base_value: float):
        """
        Sets the widget's value from an application base unit value.
        """
        if self._is_updating:
            return

        self.update_format_and_bounds()

        if not self._unit:
            return

        display_value = self._unit.from_base(base_value)
        self._is_updating = True
        self.spin_row.set_value(display_value)
        self._is_updating = False

    def get_value_in_base_units(self) -> int:
        """
        Gets the widget's current display value and converts it to base units.
        """
        if not self._unit:
            return get_spinrow_int(self.spin_row)

        display_value = self.spin_row.get_value()
        base_value = self._unit.to_base(display_value)
        return int(round(base_value))
