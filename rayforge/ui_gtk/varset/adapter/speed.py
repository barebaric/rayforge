from gettext import gettext as _
from typing import Any, Optional, Tuple

from gi.repository import Adw, Gtk

from ....context import get_context
from ....core.varset import Var
from ...shared.unit_spin_row import UnitSpinRowHelper
from .base import RowAdapter, escape_title


class SpeedRowAdapter(RowAdapter):
    """
    Adapts an Adw.SpinRow for speed values with unit conversion.

    Values are always read/written in base units (mm/min).
    """

    def __init__(self, helper: UnitSpinRowHelper) -> None:
        self._helper = helper

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "SpeedRowAdapter"]:
        machine = get_context().machine
        max_speed = machine.max_cut_speed if machine else 3000

        initial_val = getattr(var, target_property)
        min_val = getattr(var, "min_val", None) or 0

        adj = Gtk.Adjustment(
            value=int(initial_val) if initial_val is not None else 0,
            lower=min_val,
            upper=max_speed,
            step_increment=10,
            page_increment=100,
        )
        row = Adw.SpinRow(
            adjustment=adj,
            title=escape_title(var.label),
            subtitle=_("Max: {max_speed}"),
        )
        helper = UnitSpinRowHelper(
            spin_row=row,
            quantity="speed",
            max_value_in_base=max_speed,
        )
        return row, cls(helper)

    def get_value(self) -> Optional[Any]:
        return int(self._helper.get_value_in_base_units())

    def set_value(self, value: Any) -> None:
        self._helper.set_value_in_base_units(value)
