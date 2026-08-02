"""Generic spin row for one numeric step attribute."""

from typing import TYPE_CHECKING, Any, Optional

from gi.repository import Adw, Gtk

from rayforge.ui_gtk.shared.adwfix import get_spinrow_float, get_spinrow_int

from .step_row import DebouncedMixin, StepRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


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
    ):
        self._adj = Gtk.Adjustment(
            lower=lower,
            upper=upper,
            step_increment=step_inc,
            page_increment=step_inc * 10,
        )
        self.is_int = is_int
        self._digits = digits
        self._title = title
        self._subtitle = subtitle
        DebouncedMixin.__init__(self)
        StepRow.__init__(self, editor, step)
        self.attr = attr
        self.widget.connect("changed", self._on_changed)
        self._sync_from_step()
        self._sync_dependencies()

    def build_widget(self) -> Adw.SpinRow:
        if self._subtitle:
            return Adw.SpinRow(
                title=self._title,
                subtitle=self._subtitle,
                adjustment=self._adj,
                digits=self._digits,
            )
        return Adw.SpinRow(
            title=self._title,
            adjustment=self._adj,
            digits=self._digits,
        )

    def _on_changed(self, row):
        if self._syncing:
            return
        value = (
            get_spinrow_int(self.widget)
            if self.is_int
            else get_spinrow_float(self.widget)
        )
        self._debounced(self.commit, value)

    def set_widget_value(self, value):
        if value is None:
            return
        adj = self._adj
        target = float(value)
        if abs(adj.get_value() - target) > 1e-9:
            adj.set_value(target)

    def set_range(self, lower: float, upper: float):
        if (
            abs(self._adj.get_lower() - lower) > 1e-9
            or abs(self._adj.get_upper() - upper) > 1e-9
        ):
            self._adj.set_lower(lower)
            self._adj.set_upper(upper)
