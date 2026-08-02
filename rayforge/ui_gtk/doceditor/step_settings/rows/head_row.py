"""Laser head selection row widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from blinker import Signal
from gi.repository import Adw, Gtk

from rayforge.core.capability import MachineCapability

from .step_row import StepRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class HeadRow(StepRow):
    """A combo row for the base ``Step.selected_head_uid`` attribute.

    The row emits ``head_changed`` instead of committing directly:
    the head change is a transaction (kerf sync, PWM defaults) that
    is domain-specific, so the owning settings widget performs it.
    """

    def __init__(self, editor: "DocEditor", step: Any):
        machine = getattr(editor.context, "machine", None)
        heads = machine.heads if machine else []
        self._heads = [
            h for h in heads if h.machine_capability is MachineCapability.LASER
        ]
        self._machine = machine
        self.head_changed = Signal()
        StepRow.__init__(self, editor, step)
        self.attr = "selected_head_uid"
        self.set_visible(machine is not None)
        self.widget.connect("notify::selected", self._on_selected)
        self._sync_from_step()
        self._sync_dependencies()

    def build_widget(self) -> Adw.ComboRow:
        labels = [_("None")] + [h.name for h in self._heads]
        return Adw.ComboRow(
            title=_("Laser Head"),
            model=Gtk.StringList.new(labels),
        )

    def _on_selected(self, row, pspec):
        idx = self.widget.get_selected()
        if idx == Gtk.INVALID_LIST_POSITION:
            return
        uid = None if idx == 0 else self._heads[idx - 1].uid
        if uid == getattr(self.step, "selected_head_uid", None):
            return
        self.head_changed.send(self, head_uid=uid)

    def set_widget_value(self, value):
        idx = 0
        for i, head in enumerate(self._heads, start=1):
            if head.uid == value:
                idx = i
                break
        if self.widget.get_selected() != idx:
            self.widget.set_selected(idx)
