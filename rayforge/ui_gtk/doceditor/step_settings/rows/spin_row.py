"""Generic spin row for one numeric step attribute."""

from typing import TYPE_CHECKING, Any, Optional

from rayforge.ui_gtk.shared import pref_rows
from rayforge.ui_gtk.shared.pref_rows import (
    AccelerationSpinRow,
    LengthSpinRow,
    SpeedSpinRow,
)

from .step_row import DebouncedMixin, StepRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

_UNIT_ROW_CLASSES = {
    "length": LengthSpinRow,
    "speed": SpeedSpinRow,
    "acceleration": AccelerationSpinRow,
}


class SpinRow(DebouncedMixin, StepRow):
    """A spin row bound to one numeric step attribute."""

    def __init__(
        self,
        editor: "DocEditor",
        step: Any,
        attr: str,
        title: str,
        subtitle: Optional[str],
        lower: float,
        upper: float,
        step_inc: float,
        digits: int,
        is_int: bool = False,
        quantity: Optional[str] = None,
    ):
        self.is_int = is_int
        self._digits = digits
        self._title = title
        self._subtitle = subtitle
        self.quantity = quantity
        self._lower = lower
        self._upper = upper
        self._step_inc = step_inc
        DebouncedMixin.__init__(self)
        StepRow.__init__(self, editor, step)
        self.attr = attr
        self.widget.value_changed.connect(self._on_changed)
        self._sync_from_step()
        self._sync_dependencies()

    def build_widget(self):
        if self.quantity in _UNIT_ROW_CLASSES:
            cls = _UNIT_ROW_CLASSES[self.quantity]
            return cls(
                self._title,
                self._subtitle,
                lower=self._lower,
                upper=self._upper,
                step_increment=self._step_inc,
                digits=self._digits,
            )
        return pref_rows.SpinRow(
            self._title,
            self._subtitle,
            lower=self._lower,
            upper=self._upper,
            step_increment=self._step_inc,
            digits=self._digits,
        )

    def _on_changed(self, *args):
        if self._syncing:
            return
        if self.quantity:
            value = self.widget.get_value_in_base_units()
            if self.is_int:
                value = int(round(value))
        else:
            value = (
                self.widget.get_int_value()
                if self.is_int
                else self.widget.get_value()
            )
        self._debounced(self.commit, value)

    def set_widget_value(self, value):
        if value is None:
            return
        if self.quantity:
            self.widget.set_value_in_base_units(float(value))
            return
        target = float(value)
        if abs(self.widget.get_value() - target) > 1e-9:
            self.widget.set_value(target)

    def set_range(self, lower: float, upper: float):
        adj = self.widget.get_adjustment()
        if (
            abs(adj.get_lower() - lower) > 1e-9
            or abs(adj.get_upper() - upper) > 1e-9
        ):
            self.widget.set_range(lower, upper)
