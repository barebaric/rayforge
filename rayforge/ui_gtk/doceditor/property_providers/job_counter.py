import logging
from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Gtk

from ....core.item import DocItem
from ....core.job_counter import COUNTER_NAME
from ....core.workpiece import WorkPiece
from ...icons import get_icon
from ...shared.pref_rows import SpinRow
from .base import PropertyProvider

if TYPE_CHECKING:
    from ....doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class JobCounterPropertyProvider(PropertyProvider):
    """
    Shows the variable offset of the selected workpieces, with a
    shortcut to the counter settings.
    """

    priority = 30
    separate_group = True
    group_title = _("Job Counter")

    def can_handle(self, items: list[DocItem]) -> bool:
        return bool(items) and all(isinstance(i, WorkPiece) for i in items)

    def create_widgets(self) -> list[Gtk.Widget]:
        self.offset_row = SpinRow(
            _("Variable Offset"),
            lower=0,
            upper=100_000,
            numeric=True,
        )
        self.offset_row.set_tooltip_text(
            _(
                "This copy shows the values this many copies after the "
                "current ones"
            )
        )
        self.offset_row.value_changed.connect(self._on_offset_changed)

        self.settings_button = Gtk.Button(
            child=get_icon("settings-symbolic"),
            valign=Gtk.Align.CENTER,
            tooltip_text=_("Job Counter Settings"),
            action_name="win.job-counter",
        )
        self.offset_row.add_suffix(self.settings_button)
        self._rows = [self.offset_row]
        return self._rows

    def update_widgets(self, editor: "DocEditor", items: list[DocItem]):
        self.editor = editor
        self.items = items
        workpieces = [i for i in items if isinstance(i, WorkPiece)]
        if not workpieces:
            return
        first = workpieces[0]
        self._in_update = True
        try:
            self.offset_row.set_value(first.variable_offset)
        finally:
            self._in_update = False

        if len(workpieces) > 1:
            self.offset_row.set_subtitle(_("Applies to all selected items"))
            return
        counter = first.template_values().get(COUNTER_NAME)
        self.offset_row.set_subtitle(
            _("Job counter: {value}").format(value=counter)
        )

    def _on_offset_changed(self, row: SpinRow) -> None:
        if self._in_update:
            return
        workpieces = [i for i in self.items if isinstance(i, WorkPiece)]
        self.editor.job_counter.set_variable_offset(
            workpieces, row.get_int_value()
        )
