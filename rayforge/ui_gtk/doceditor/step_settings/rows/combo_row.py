"""Generic combo row for an enum-like step attribute."""

from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

from gi.repository import Adw, Gtk

from .step_row import StepRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class ComboRow(StepRow):
    """A combo row bound to an enum-like step attribute.

    ``choices`` is a sequence of ``(label, value)`` pairs where
    ``value`` is what is stored on the step.
    """

    def __init__(
        self,
        editor: "DocEditor",
        step: Any,
        attr: str,
        title: str,
        choices: Sequence[Tuple[str, Any]],
        subtitle: Optional[str] = None,
    ):
        self._choices = list(choices)
        self._labels = [label for label, _ in choices]
        self._title = title
        self._subtitle = subtitle
        StepRow.__init__(self, editor, step)
        self.attr = attr
        self.widget.connect("notify::selected", self._on_selected)
        self._sync_from_step()
        self._sync_dependencies()

    def build_widget(self) -> Adw.ComboRow:
        model = Gtk.StringList.new(self._labels)
        if self._subtitle:
            return Adw.ComboRow(
                title=self._title,
                subtitle=self._subtitle,
                model=model,
            )
        return Adw.ComboRow(title=self._title, model=model)

    def _on_selected(self, row, pspec):
        if self._syncing:
            return
        idx = self.widget.get_selected()
        if idx == Gtk.INVALID_LIST_POSITION:
            return
        self.commit(self._choices[idx][1])

    def set_widget_value(self, value):
        for i, (label, stored) in enumerate(self._choices):
            if stored == value:
                if self.widget.get_selected() != i:
                    self.widget.set_selected(i)
                return
