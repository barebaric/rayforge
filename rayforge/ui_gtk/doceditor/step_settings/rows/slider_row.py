"""Generic slider row for one numeric step attribute."""

import locale
from typing import TYPE_CHECKING, Any, Optional

from gi.repository import Adw, Gtk

from rayforge.ui_gtk.shared.slider import create_slider

from .step_row import DebouncedMixin, StepRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class SliderRow(DebouncedMixin, StepRow):
    """A slider row bound to one numeric step attribute."""

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
    ):
        self._adj = Gtk.Adjustment(
            lower=lower,
            upper=upper,
            step_increment=step_inc,
            page_increment=step_inc * 10,
        )
        self._digits = digits
        self._title = title
        self._subtitle = subtitle
        DebouncedMixin.__init__(self)
        StepRow.__init__(self, editor, step)
        self.attr = attr
        self._scale.connect("value-changed", self._on_scale)
        self._sync_from_step()
        self._sync_dependencies()

    def build_widget(self) -> Adw.ActionRow:
        if self._subtitle:
            row = Adw.ActionRow(title=self._title, subtitle=self._subtitle)
        else:
            row = Adw.ActionRow(title=self._title)
        value_text = self._format(self._adj.get_value())
        self._value_label = Gtk.Label(label=value_text)
        self._value_label.add_css_class("dim-label")
        self._value_label.set_width_chars(6)
        self._scale = create_slider(
            adjustment=self._adj,
            digits=self._digits,
            draw_value=False,
        )
        row.add_suffix(self._value_label)
        row.add_suffix(self._scale)
        return row

    def _format(self, value: float) -> str:
        return locale.format_string(f"%.{self._digits}f", value)

    def _on_scale(self, scale):
        self._value_label.set_text(self._format(self._adj.get_value()))
        if self._syncing:
            return
        self._debounced(self.commit, self._adj.get_value())

    def set_widget_value(self, value):
        if value is None:
            return
        target = float(value)
        if abs(self._adj.get_value() - target) > 1e-9:
            self._adj.set_value(target)
            self._value_label.set_text(self._format(target))

    def set_range(self, lower: float, upper: float):
        if (
            abs(self._adj.get_lower() - lower) > 1e-9
            or abs(self._adj.get_upper() - upper) > 1e-9
        ):
            self._adj.set_lower(lower)
            self._adj.set_upper(upper)
