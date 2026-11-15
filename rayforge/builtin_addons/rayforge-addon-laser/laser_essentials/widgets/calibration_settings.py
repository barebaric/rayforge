"""
The settings of one calibration test, as an edit loop that can be
attached to any widget that hosts it.

The layer settings dialog shows them in its Test Settings section.
Every change rebuilds the test after a short pause, as one undo entry.
"""

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, GLib, Gtk

from rayforge.ui_gtk.mainwindow import MainWindow
from rayforge.ui_gtk.varset.varsetwidget import VarSetWidget

from ..calibration import GENERATORS, CalibrationGenerator
from ..calibration.record import CalibrationRecord
from ..commands import CalibrationCmd

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

logger = logging.getLogger(__name__)

DEBOUNCE_DELAY_MS = 300


class CalibrationSettingsController:
    """Shows and applies the parameters of one calibration test."""

    def __init__(
        self,
        editor: "DocEditor",
        record: CalibrationRecord,
    ):
        self.editor = editor
        self.record = record
        self._in_update = False
        self._pending_keys: set[str] = set()
        self._timer_id = 0

        self.help_row = _text_row("dim-label")
        self.warning_row = _text_row("warning")
        self.error_row = _text_row("error")
        self.note_row = _text_row("dim-label")

        self.group = VarSetWidget()
        self.group.set_title(_("Test Settings"))
        for row in (self.help_row, self.warning_row, self.error_row):
            self.group.add(row)
        generator = self.generator
        if generator is not None:
            params = CalibrationCmd(editor).params(record)
            self.group.populate(
                generator.settings_varset(params, editor.context)
            )
        self.group.add(self.note_row)
        self.group.data_changed.connect(self._on_data_changed)
        self.refresh()

    @property
    def generator(self) -> CalibrationGenerator | None:
        return GENERATORS.get(self.record.generator_id)

    def widgets(self) -> list[Adw.PreferencesGroup]:
        return [self.group]

    def close(self):
        """Stops following the edits when the hosting UI goes away."""
        if self._timer_id:
            GLib.source_remove(self._timer_id)
            self._timer_id = 0
        self.group.data_changed.disconnect(self._on_data_changed)

    def refresh(self, params: dict[str, Any] | None = None):
        """
        Shows *params*, or the parameters of the test as they are now
        when none are given.
        """
        generator = self.generator
        if generator is None:
            return
        self._in_update = True
        try:
            if params is None:
                params = CalibrationCmd(self.editor).params(self.record)
            keys = set(self.group.keys())
            self.group.set_values(
                {k: v for k, v in params.items() if k in keys}
            )
            for key, (labels, subtitles) in generator.row_labels(
                params
            ).items():
                adapter = self.group.adapter_for(key)
                if adapter is not None:
                    adapter.set_item_labels(labels)
                    adapter.set_item_subtitles(subtitles)
            _show_text(self.help_row, generator.help_text(params))
            _show_text(
                self.warning_row,
                generator.warning(params, self.editor.context),
            )
            self.error_row.set_visible(False)
            _show_text(self.note_row, _rebuild_note(generator))
        finally:
            self._in_update = False

    def _on_data_changed(self, sender: VarSetWidget, key: str):
        if self._in_update:
            return
        self._pending_keys.add(key)
        if self._timer_id:
            GLib.source_remove(self._timer_id)
        self._timer_id = GLib.timeout_add(DEBOUNCE_DELAY_MS, self._on_timer)

    def _on_timer(self) -> bool:
        self._timer_id = 0
        self.apply_now()
        return GLib.SOURCE_REMOVE

    def apply_now(self):
        """Rebuilds the test from the values in the rows."""
        if self._timer_id:
            GLib.source_remove(self._timer_id)
            self._timer_id = 0
        keys = set(self._pending_keys)
        self._pending_keys.clear()
        generator = self.generator
        if not keys or generator is None:
            return
        cmd = CalibrationCmd(self.editor)
        values = self.group.get_values()
        params = cmd.params(self.record)
        params.update({key: values[key] for key in keys if key in values})
        params = generator.apply_change(params, keys, self.editor.context)
        self.refresh(params)
        try:
            workpieces = cmd.update_params(self.record, params)
        except ValueError as e:
            _show_text(self.error_row, str(e))
            self._pending_keys |= keys
            return
        self.error_row.set_visible(False)
        self._reselect(workpieces)

    def _reselect(self, workpieces):
        """
        Selects the rebuilt test on the canvas. The old workpieces are
        gone, so the selection was cleared; it is restored before the
        panel is drawn again.
        """
        window = self._main_window()
        if window is not None:
            window.surface.select_items(workpieces)

    def _main_window(self) -> MainWindow | None:
        root = self.group.get_root()
        if isinstance(root, MainWindow):
            return root
        if isinstance(root, Gtk.Window):
            parent = root.get_transient_for()
            if isinstance(parent, MainWindow):
                return parent
        return None


def _text_row(css_class: str) -> Adw.ActionRow:
    row = Adw.ActionRow(activatable=False)
    row.set_title_lines(0)
    row.add_css_class(css_class)
    row.set_visible(False)
    return row


def _show_text(row: Adw.ActionRow, text: str):
    row.set_title(GLib.markup_escape_text(text))
    row.set_visible(bool(text))


def _rebuild_note(generator: CalibrationGenerator) -> str:
    if generator.keeps_steps:
        return _(
            "Changing a setting rebuilds the test. The other settings of "
            "its operation stay."
        )
    return _(
        "Changing a setting rebuilds the test and replaces changes made "
        "to its operations."
    )
