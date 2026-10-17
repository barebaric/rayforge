import logging
from gettext import gettext as _
from pathlib import Path
from typing import TYPE_CHECKING

from gi.repository import Adw, Gio, GLib, Gtk

from ...core.variable_text import CsvMerge
from ..shared.pref_rows import SpinRow

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)

_ROW_LIMIT = 999_999_999
_PREVIEW_ROWS_MAX = 3


def format_rows(merge: CsvMerge, rows: list[int]) -> str:
    """A short text for the rows the next job uses."""
    if not merge.is_loaded:
        return _("No CSV file loaded")
    if merge.is_exhausted:
        return _("All rows have been used")
    shown = []
    for number in rows[:_PREVIEW_ROWS_MAX]:
        if number > merge.row_count:
            shown.append(_("{row} (empty)").format(row=number))
            continue
        first_cell = merge.rows[number - 1][0] if merge.header else ""
        shown.append(f"{number}: {first_cell}")
    if len(rows) > _PREVIEW_ROWS_MAX:
        shown.append(_("… {count} copies").format(count=len(rows)))
    return ", ".join(shown)


class CsvMergeGroup(Adw.PreferencesGroup):
    """The CSV merge settings of the Variable Text dialog."""

    def __init__(self, editor: "DocEditor"):
        super().__init__(
            title=_("CSV Merge"),
            description=_(
                'Use {csv["Name"]} or {csv[0]} for a column of the '
                "current row, and {row} for its number."
            ),
        )
        self.editor = editor
        self._in_update = False

        self.file_row = Adw.ActionRow(title=_("CSV File"))
        self.choose_button = Gtk.Button(
            label=_("Choose…"), valign=Gtk.Align.CENTER
        )
        self.reload_button = Gtk.Button(
            icon_name="view-refresh-symbolic", valign=Gtk.Align.CENTER
        )
        self.reload_button.set_tooltip_text(_("Read the file again"))
        self.file_row.add_suffix(self.reload_button)
        self.file_row.add_suffix(self.choose_button)
        self.columns_row = Adw.ActionRow(title=_("Columns"))
        self.columns_row.add_css_class("property")
        self.current_row = SpinRow(
            _("Current Row"),
            _("The row the next job starts with"),
            lower=1,
            upper=_ROW_LIMIT,
            numeric=True,
        )
        self.auto_advance_row = Adw.SwitchRow(
            title=_("Advance After Each Job"),
            subtitle=_("Stops after the last row"),
        )
        self.preview_row = Adw.ActionRow(title=_("Next Job"))
        self.preview_row.add_css_class("property")
        self.copies_row = Adw.ActionRow(
            title=_("Copies Needed"),
            subtitle=_("Copies that cover the remaining rows in one job"),
        )
        self.copies_label = Gtk.Label()
        self.copies_row.add_suffix(self.copies_label)

        for row in (
            self.file_row,
            self.columns_row,
            self.current_row,
            self.auto_advance_row,
            self.preview_row,
            self.copies_row,
        ):
            self.add(row)
        self.add(self._build_buttons())

        self.choose_button.connect("clicked", self._on_choose_clicked)
        self.reload_button.connect("clicked", self._on_reload_clicked)
        self.current_row.value_changed.connect(self._on_current_changed)
        self.auto_advance_row.connect(
            "notify::active", self._on_auto_advance_changed
        )

    def _build_buttons(self) -> Gtk.Box:
        box = Gtk.Box(spacing=6, halign=Gtk.Align.END)
        box.set_margin_top(12)
        self.first_button = Gtk.Button(label=_("First"))
        self.first_button.set_tooltip_text(_("Go back to the first row"))
        self.previous_button = Gtk.Button(label=_("Previous"))
        self.previous_button.set_tooltip_text(
            _("Go back by the rows of one job")
        )
        self.next_button = Gtk.Button(label=_("Next"))
        self.next_button.set_tooltip_text(
            _("Skip ahead by the rows of one job")
        )
        commands = self.editor.variable_text
        self.first_button.connect("clicked", lambda b: commands.first_row())
        self.previous_button.connect(
            "clicked", lambda b: commands.previous_row()
        )
        self.next_button.connect("clicked", lambda b: commands.next_row())
        for button in (
            self.first_button,
            self.previous_button,
            self.next_button,
        ):
            box.append(button)
        return box

    def update(self) -> None:
        """Shows the document's CSV merge."""
        merge = self.editor.variable_text.csv
        self._in_update = True
        try:
            self.file_row.set_subtitle(
                GLib.markup_escape_text(merge.path or _("No file linked"))
            )
            self.reload_button.set_sensitive(bool(merge.path))
            self.columns_row.set_subtitle(
                GLib.markup_escape_text(
                    _("{columns} ({rows} rows)").format(
                        columns=", ".join(merge.header) or "-",
                        rows=merge.row_count,
                    )
                )
            )
            self.current_row.set_value(merge.current)
            self.auto_advance_row.set_active(merge.auto_advance)
            for widget in (
                self.current_row,
                self.auto_advance_row,
                self.first_button,
                self.previous_button,
                self.next_button,
            ):
                widget.set_sensitive(merge.is_loaded)
        finally:
            self._in_update = False
        self.update_preview()

    def update_preview(self) -> None:
        """Shows the rows the next job uses."""
        commands = self.editor.variable_text
        merge = commands.csv
        self.preview_row.set_subtitle(
            GLib.markup_escape_text(
                format_rows(merge, commands.preview_rows())
            )
        )
        self.copies_label.set_label(str(commands.copies_needed()))

    def _on_current_changed(self, row: SpinRow) -> None:
        if self._in_update:
            return
        commands = self.editor.variable_text
        commands.set_csv(commands.csv.with_current(row.get_int_value()))

    def _on_auto_advance_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        if self._in_update:
            return
        commands = self.editor.variable_text
        merge = commands.csv
        commands.set_csv(
            CsvMerge(
                header=merge.header,
                rows=merge.rows,
                path=merge.path,
                current=merge.current,
                auto_advance=row.get_active(),
            )
        )

    def _on_reload_clicked(self, button: Gtk.Button) -> None:
        self.editor.variable_text.reload_csv()

    def _on_choose_clicked(self, button: Gtk.Button) -> None:
        dialog = Gtk.FileDialog.new()
        dialog.set_title(_("Choose a CSV File"))
        csv_filter = Gtk.FileFilter()
        csv_filter.set_name(_("CSV files"))
        csv_filter.add_pattern("*.csv")
        csv_filter.add_pattern("*.txt")
        csv_filter.add_mime_type("text/csv")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(csv_filter)
        dialog.set_filters(filters)
        path = self.editor.variable_text.csv.path
        if path and Path(path).parent.is_dir():
            folder = Gio.File.new_for_path(str(Path(path).parent))
            dialog.set_initial_folder(folder)
        root = self.get_root()
        parent = root if isinstance(root, Gtk.Window) else None
        dialog.open(parent, None, self._on_file_chosen)

    def _on_file_chosen(self, dialog: Gtk.FileDialog, result) -> None:
        try:
            file = dialog.open_finish(result)
        except GLib.Error as e:
            logger.debug(f"CSV file choice cancelled: {e.message}")
            return
        path = file.get_path() if file is not None else None
        if path:
            self.editor.variable_text.load_csv(path)
