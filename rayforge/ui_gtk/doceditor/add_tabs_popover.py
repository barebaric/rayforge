from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List
from gi.repository import Gtk, Adw

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class AddTabsPopover(Gtk.Popover):
    def __init__(
        self,
        editor: DocEditor,
        workpieces: List[WorkPiece],
    ):
        super().__init__()
        self.editor = editor
        self.workpieces = workpieces
        self._in_update = False

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(12)
        content_box.set_margin_start(12)
        content_box.set_margin_end(12)
        self.set_child(content_box)

        rows_container = Gtk.ListBox()
        rows_container.set_selection_mode(Gtk.SelectionMode.NONE)
        rows_container.add_css_class("boxed-list")
        content_box.append(rows_container)

        self.tab_count_row = Adw.SpinRow(
            title=_("Number of Tabs"),
            adjustment=Gtk.Adjustment.new(4, 1, 1000, 1, 10, 0),
            digits=0,
        )
        rows_container.append(self.tab_count_row)

        self.tab_width_row = Adw.SpinRow(
            title=_("Tab Width (mm)"),
            adjustment=Gtk.Adjustment.new(2.0, 0.1, 100, 0.1, 1, 0),
            digits=2,
        )
        rows_container.append(self.tab_width_row)

        # Use the first workpiece to set initial values
        first_workpiece = self.workpieces[0]

        self._in_update = True
        initial_count = len(first_workpiece.tabs)
        if initial_count > 0:
            self.tab_count_row.set_value(initial_count)
            self.tab_width_row.set_value(first_workpiece.tabs[0].width)
        else:
            self.tab_count_row.set_value(4)
            self.tab_width_row.set_value(2.0)
        self._in_update = False

        # Connect signals for live updates
        self.tab_count_row.connect("notify::value", self._on_value_changed)
        self.tab_width_row.connect("notify::value", self._on_value_changed)

        # Trigger the initial command to set the default tabs
        self._on_value_changed(None, None)

    def _on_value_changed(self, spin_row, GParamSpec):
        if self._in_update:
            return

        count = int(self.tab_count_row.get_value())
        width = self.tab_width_row.get_value()

        # Group all changes into a single undoable transaction.
        # This is the correct way to batch changes that should be undone
        # together. However, for live updates where each tweak should be
        # undoable, we execute commands directly.
        with self.editor.history_manager.transaction(
            _("Adjust Equidistant Tabs")
        ):
            for workpiece in self.workpieces:
                if (
                    not workpiece.layer
                    or not workpiece.layer.workflow
                    or not workpiece.layer.workflow.steps
                ):
                    continue

                self.editor.tab.add_tabs(
                    workpiece=workpiece,
                    count=count,
                    width=width,
                    strategy="equidistant",
                )
