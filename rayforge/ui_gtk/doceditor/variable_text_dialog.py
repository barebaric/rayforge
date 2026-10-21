import logging
from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw, Gtk

from ...core.variable_text import SerialSettings, VariableText
from ..shared.patched_dialog_window import PatchedDialogWindow
from ..shared.pref_rows import SpinRow
from .variable_text_csv_group import CsvMergeGroup

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)

_PREVIEW_LIST_MAX = 5
_INT_LIMIT = 999_999_999


def format_preview(values: list[int]) -> str:
    """A short text for the serial numbers of the next job."""
    if len(values) <= _PREVIEW_LIST_MAX:
        return ", ".join(str(v) for v in values)
    return _("{first} … {last} ({count} copies)").format(
        first=values[0], last=values[-1], count=len(values)
    )


class VariableTextDialog(PatchedDialogWindow):
    """
    A non-modal window for the document's serial number and CSV merge:
    their settings, navigation buttons and a preview of the next job.
    """

    def __init__(self, parent: Gtk.Window, editor: "DocEditor"):
        super().__init__(transient_for=parent)
        self.editor = editor
        self._doc: Doc = editor.doc
        self._in_update = False

        self.set_title(_("Variable Text"))
        self.set_default_size(520, -1)
        self.set_modal(False)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main_box.append(Adw.HeaderBar())
        self.set_content(main_box)

        page = Adw.PreferencesPage()
        page.set_vexpand(True)
        main_box.append(page)

        page.add(self._build_serial_group())
        page.add(self._build_job_group())
        self.csv_group = CsvMergeGroup(editor)
        page.add(self.csv_group)

        self._doc.variable_text.changed.connect(self._on_model_changed)
        self._doc.descendant_updated.connect(self._on_doc_items_changed)
        self._doc.descendant_added.connect(self._on_doc_items_changed)
        self._doc.descendant_removed.connect(self._on_doc_items_changed)
        self.connect("destroy", self._on_destroy)
        self._update_from_model()

    def _build_serial_group(self) -> Adw.PreferencesGroup:
        group = Adw.PreferencesGroup(
            title=_("Serial Number"),
            description=_(
                "Use {serial} in a text box, e.g. No. {serial:04d} "
                "for four digits with leading zeros."
            ),
        )
        self.current_row = SpinRow(
            _("Current Value"),
            _("The number the next job starts with"),
            lower=-_INT_LIMIT,
            upper=_INT_LIMIT,
            numeric=True,
        )
        self.start_row = SpinRow(
            _("Start Value"),
            _("Reset returns here; the range wraps back to it"),
            lower=-_INT_LIMIT,
            upper=_INT_LIMIT,
            numeric=True,
        )
        self.step_row = SpinRow(
            _("Advance By"),
            _("Increment per copy and per job"),
            lower=0,
            upper=_INT_LIMIT,
            numeric=True,
        )
        self.limit_row = Adw.SwitchRow(
            title=_("Wrap at End Value"),
            subtitle=_("Continue at the start value after the end value"),
        )
        self.end_row = SpinRow(
            _("End Value"),
            lower=-_INT_LIMIT,
            upper=_INT_LIMIT,
            numeric=True,
        )
        self.auto_advance_row = Adw.SwitchRow(
            title=_("Advance After Each Job"),
            subtitle=_("When a job sent to the machine has finished"),
        )
        self.export_row = Adw.SwitchRow(
            title=_("Advance After G-code Export"),
            subtitle=_("For jobs that run from a memory card"),
        )

        for row in (self.current_row, self.start_row, self.step_row):
            row.value_changed.connect(self._on_value_row_changed)
        self.end_row.value_changed.connect(self._on_value_row_changed)
        for switch in (self.limit_row, self.auto_advance_row, self.export_row):
            switch.connect("notify::active", self._on_switch_changed)

        for row in (
            self.current_row,
            self.start_row,
            self.step_row,
            self.limit_row,
            self.end_row,
            self.auto_advance_row,
            self.export_row,
        ):
            group.add(row)
        return group

    def _build_job_group(self) -> Adw.PreferencesGroup:
        group = Adw.PreferencesGroup(title=_("Next Job"))
        self.preview_row = Adw.ActionRow(title=_("Serial Numbers"))
        self.preview_row.add_css_class("property")
        group.add(self.preview_row)

        button_box = Gtk.Box(spacing=6, halign=Gtk.Align.END)
        button_box.set_margin_top(12)
        self.previous_button = Gtk.Button(label=_("Previous"))
        self.previous_button.set_tooltip_text(
            _("Go back by the numbers of one job")
        )
        self.next_button = Gtk.Button(label=_("Next"))
        self.next_button.set_tooltip_text(
            _("Skip ahead by the numbers of one job")
        )
        self.reset_button = Gtk.Button(label=_("Reset"))
        self.reset_button.set_tooltip_text(_("Go back to the start value"))
        self.previous_button.connect("clicked", self._on_previous_clicked)
        self.next_button.connect("clicked", self._on_next_clicked)
        self.reset_button.connect("clicked", self._on_reset_clicked)
        for button in (
            self.reset_button,
            self.previous_button,
            self.next_button,
        ):
            button_box.append(button)
        group.add(button_box)
        return group

    def _settings_from_widgets(self) -> SerialSettings:
        start = self.start_row.get_int_value()
        end = self.end_row.get_int_value()
        limited = self.limit_row.get_active() and end >= start
        return SerialSettings(
            current=self.current_row.get_int_value(),
            start=start,
            step=self.step_row.get_int_value(),
            end=end if limited else None,
            auto_advance=self.auto_advance_row.get_active(),
            advance_on_export=self.export_row.get_active(),
        )

    def _commit(self) -> None:
        if self._in_update:
            return
        self.editor.variable_text.set_serial(self._settings_from_widgets())

    def _on_value_row_changed(self, row: SpinRow) -> None:
        self._commit()

    def _on_switch_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        self.end_row.set_sensitive(self.limit_row.get_active())
        self._commit()

    def _on_previous_clicked(self, button: Gtk.Button) -> None:
        self.editor.variable_text.previous_serial()

    def _on_next_clicked(self, button: Gtk.Button) -> None:
        self.editor.variable_text.next_serial()

    def _on_reset_clicked(self, button: Gtk.Button) -> None:
        self.editor.variable_text.reset_serial()

    def _on_model_changed(self, sender: VariableText) -> None:
        self._update_from_model()

    def _on_doc_items_changed(self, sender, **kwargs) -> None:
        self._update_preview()
        self.csv_group.update_preview()

    def _update_from_model(self) -> None:
        serial = self._doc.variable_text.serial
        self._in_update = True
        try:
            self.current_row.set_value(serial.current)
            self.start_row.set_value(serial.start)
            self.step_row.set_value(serial.step)
            self.limit_row.set_active(serial.end is not None)
            if serial.end is not None:
                self.end_row.set_value(serial.end)
            elif self.end_row.get_int_value() < serial.start:
                self.end_row.set_value(serial.start)
            self.end_row.set_sensitive(serial.end is not None)
            self.auto_advance_row.set_active(serial.auto_advance)
            self.export_row.set_active(serial.advance_on_export)
        finally:
            self._in_update = False
        self._update_preview()
        self.csv_group.update()

    def _update_preview(self) -> None:
        values = self.editor.variable_text.preview_serial_values()
        self.preview_row.set_subtitle(format_preview(values))

    def _on_destroy(self, widget) -> None:
        variable_text = self._doc.variable_text
        variable_text.changed.disconnect(self._on_model_changed)
        self._doc.descendant_updated.disconnect(self._on_doc_items_changed)
        self._doc.descendant_added.disconnect(self._on_doc_items_changed)
        self._doc.descendant_removed.disconnect(self._on_doc_items_changed)
