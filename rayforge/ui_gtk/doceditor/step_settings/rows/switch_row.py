"""Generic switch row for a boolean step attribute."""

from typing import TYPE_CHECKING, Any

from gi.repository import Adw

from .step_row import StepRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class SwitchRow(StepRow):
    """A switch row bound to a boolean step attribute."""

    def __init__(
        self,
        editor: "DocEditor",
        step: Any,
        attr: str,
        title: str,
        subtitle: str | None = None,
    ):
        self._title = title
        self._subtitle = subtitle
        StepRow.__init__(self, editor, step)
        self.attr = attr
        self.widget.connect("notify::active", self._on_active)
        self._sync_from_step()
        self._sync_dependencies()

    def build_widget(self) -> Adw.SwitchRow:
        if self._subtitle:
            return Adw.SwitchRow(title=self._title, subtitle=self._subtitle)
        return Adw.SwitchRow(title=self._title)

    def _on_active(self, row, pspec):
        if self._syncing:
            return
        self.commit(self.widget.get_active())

    def set_widget_value(self, value):
        if value is not None and self.widget.get_active() != bool(value):
            self.widget.set_active(bool(value))
