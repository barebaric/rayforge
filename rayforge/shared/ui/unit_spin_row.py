import logging
from typing import Optional
from gi.repository import Adw, Gtk
from blinker import Signal
from ...context import get_context
from ..units.definitions import Unit, get_unit, get_units_for_quantity
from .adwfix import get_spinrow_float
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
        self._config_handler_id = get_context().config.changed.connect(
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
            get_context().config.changed.disconnect(self._config_handler_id)
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
        config = get_context().config
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
                + f" ({self._unit.label})"
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

    def get_value_in_base_units(self) -> float:
        """
        Gets the widget's current display value and converts it to base units.
        """
        if not self._unit:
            return get_spinrow_float(self.spin_row)

        display_value = get_spinrow_float(self.spin_row)
        base_value = self._unit.to_base(display_value)
        return float(base_value)


class UnitSelectorSpinRow:
    """
    A widget that combines a SpinRow with a unit selector dropdown.

    This widget allows users to both specify a value and choose the unit
    in a single Adw-style row, maintaining visual consistency with the
    rest of the application.
    """

    def __init__(
        self,
        quantity: str,
        title: str,
        subtitle: str = "",
        max_value_in_base: Optional[float] = None,
    ):
        self.quantity = quantity
        self.title = title
        self.subtitle = subtitle
        self._max_value_in_base = max_value_in_base
        self._is_updating = False

        # Application-level signal for value changes (in base units)
        self.changed = Signal()

        # Create the main action row
        self.row = Adw.ActionRow()
        self.row.set_title(title)
        if subtitle:
            self.row.set_subtitle(subtitle)

        # Create the spin button for value
        self.spin_button = Gtk.SpinButton()
        adjustment = Gtk.Adjustment(
            lower=0,
            upper=999,
            step_increment=1,
            page_increment=10,
        )
        self.spin_button.set_adjustment(adjustment)
        self.spin_button.set_valign(Gtk.Align.CENTER)

        # Create the unit dropdown
        self.unit_dropdown = Gtk.DropDown()
        self.unit_dropdown.set_valign(Gtk.Align.CENTER)

        # Setup the helper for unit conversion
        self.helper = UnitSpinRowHelper(
            spin_row=self._create_spin_row_wrapper(),
            quantity=quantity,
            max_value_in_base=max_value_in_base,
        )

        # Populate unit options
        self._populate_units()

        # Connect signals
        self.unit_dropdown.connect("notify::selected", self._on_unit_changed)
        self.helper.changed.connect(self._on_value_changed)

        # Add widgets to the row (unit after value)
        self.row.add_suffix(self.spin_button)
        self.row.add_suffix(self.unit_dropdown)

        # Initial update
        self._update_format_and_bounds()

    def _create_spin_row_wrapper(self):
        """
        Creates a minimal SpinRow wrapper for the UnitSpinRowHelper.
        This is a workaround since UnitSpinRowHelper expects a SpinRow.
        """
        spin_row = Adw.SpinRow()
        spin_row.set_adjustment(self.spin_button.get_adjustment())
        return spin_row

    def _populate_units(self):
        """
        Populates the unit dropdown with available units for the quantity.
        """
        units = get_units_for_quantity(self.quantity)
        if not units:
            return

        # Create string list for dropdown
        string_list = Gtk.StringList()
        for unit in units:
            string_list.append(unit.label)

        self.unit_dropdown.set_model(string_list)

        config = get_context().config
        current_unit_name = config.unit_preferences.get(self.quantity)

        for i, unit in enumerate(units):
            if unit.name == current_unit_name:
                self.unit_dropdown.set_selected(i)
                break

    def _on_unit_changed(self, dropdown, pspec):
        """Handles unit selection changes."""
        if self._is_updating:
            return

        # Get the selected unit
        selected_index = self.unit_dropdown.get_selected()
        units = get_units_for_quantity(self.quantity)

        if selected_index < len(units):
            unit = units[selected_index]

            config = get_context().config
            config.unit_preferences[self.quantity] = unit.name
            config.changed.send(config)

            # Update format and bounds
            self._update_format_and_bounds()

    def _on_value_changed(self, helper):
        """Handles value changes from the UnitSpinRowHelper."""
        if self._is_updating:
            return

        # Forward the signal
        if hasattr(self, "changed"):
            self.changed.send(self)

    def _update_format_and_bounds(self):
        """Updates the format and bounds based on the current unit."""
        self.helper.update_format_and_bounds()

        # Update subtitle with unit information
        if self.helper._unit:
            self.row.set_subtitle(
                f"{self.subtitle} ({self.helper._unit.label})"
            )

    def set_value_in_base_units(self, base_value: float):
        """Sets the widget's value from an application base unit value."""
        self.helper.set_value_in_base_units(base_value)

    def get_value_in_base_units(self) -> float:
        """
        Gets the widget's current display value and converts it to base
        units.
        """
        return self.helper.get_value_in_base_units()
