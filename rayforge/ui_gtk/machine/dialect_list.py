from typing import cast
from gi.repository import Gtk, Adw
from ...machine.models.dialect import GcodeDialect, get_available_dialects
from ...context import get_context
from ..icons import get_icon
from ..settings.preferences_group import PreferencesGroupWithButton
from .dialect_editor import DialectEditorDialog


class DialectRow(Gtk.Box):
    """A widget representing a single Dialect in a ListBox."""

    def __init__(self, dialect: GcodeDialect):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.dialect = dialect
        self.dialect_mgr = get_context().dialect_mgr
        self._setup_ui()

    def _setup_ui(self):
        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        if not self.dialect.is_custom:
            lock_icon = get_icon("lock-symbolic")
            self.append(lock_icon)

        title_label = Gtk.Label(
            label=self.dialect.label,
            halign=Gtk.Align.START,
            hexpand=True,
            xalign=0,
        )
        self.append(title_label)

        suffix_box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        self.append(suffix_box)

        if self.dialect.is_custom:
            edit_button = Gtk.Button(child=get_icon("document-edit-symbolic"))
            edit_button.add_css_class("flat")
            edit_button.connect("clicked", self._on_edit_clicked)
            suffix_box.append(edit_button)

            delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
            delete_button.add_css_class("flat")
            delete_button.connect("clicked", self._on_delete_clicked)
            suffix_box.append(delete_button)
        else:
            copy_button = Gtk.Button(child=get_icon("copy-symbolic"))
            copy_button.add_css_class("flat")
            copy_button.set_tooltip_text(_("Copy & Edit"))
            copy_button.connect("clicked", self._on_copy_edit_clicked)
            suffix_box.append(copy_button)

    def _on_copy_edit_clicked(self, button: Gtk.Button):
        parent = cast(Gtk.Window, self.get_ancestor(Gtk.Window))
        new_label = _("{label} (Copy)").format(label=self.dialect.label)
        new_dialect = self.dialect.copy_as_custom(new_label=new_label)

        dialog = DialectEditorDialog(parent, new_dialect)
        dialog.connect("close-request", self._on_new_dialect_dialog_closed)
        dialog.present()

    def _on_new_dialect_dialog_closed(self, dialog: DialectEditorDialog):
        if dialog.saved:
            self.dialect_mgr.add_dialect(dialog.dialect)

    def _on_edit_clicked(self, button: Gtk.Button):
        parent = cast(Gtk.Window, self.get_ancestor(Gtk.Window))
        dialog = DialectEditorDialog(parent, self.dialect)
        dialog.connect("close-request", self._on_edit_dialog_closed)
        dialog.present()

    def _on_edit_dialog_closed(self, dialog: DialectEditorDialog):
        if dialog.saved:
            self.dialect_mgr.update_dialect(dialog.dialect)

    def _on_delete_clicked(self, button: Gtk.Button):
        parent = cast(Gtk.Window, self.get_ancestor(Gtk.Window))
        dialog = Adw.MessageDialog(
            transient_for=parent,
            heading=_("Delete '{label}'?").format(label=self.dialect.label),
            body=_(
                "This custom dialect will be permanently removed. "
                "This action cannot be undone."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("delete", _("Delete"))
        dialog.set_response_appearance(
            "delete", Adw.ResponseAppearance.DESTRUCTIVE
        )
        dialog.connect("response", self._on_delete_response)
        dialog.present()

    def _on_delete_response(self, dialog: Adw.MessageDialog, response_id: str):
        if response_id == "delete":
            self.dialect_mgr.delete_dialect(self.dialect)


class DialectListEditor(PreferencesGroupWithButton):
    """An Adwaita widget for managing a list of G-code dialects."""

    def __init__(self, **kwargs):
        super().__init__(button_label=_("Add New Dialect"), **kwargs)
        self.dialect_mgr = get_context().dialect_mgr
        self._setup_ui()
        self.dialect_mgr.dialects_changed.connect(self._on_dialects_changed)
        self.connect("destroy", self._on_destroy)
        self._on_dialects_changed()  # Initial population

    def _setup_ui(self):
        """Configures the widget and its placeholder."""
        placeholder = Gtk.Label(
            label=_("No custom dialects configured"),
            halign=Gtk.Align.CENTER,
            margin_top=12,
            margin_bottom=12,
        )
        placeholder.add_css_class("dim-label")
        self.list_box.set_placeholder(placeholder)

    def _on_destroy(self, *args):
        self.dialect_mgr.dialects_changed.disconnect(self._on_dialects_changed)

    def _on_dialects_changed(self, sender=None, **kwargs):
        """Callback to rebuild the list when the dialect manager signals."""
        all_dialects = get_available_dialects()
        # Sort to show built-ins first, then customs alphabetically
        sorted_dialects = sorted(
            all_dialects, key=lambda d: (d.is_custom, d.label)
        )
        self.set_items(sorted_dialects)

    def create_row_widget(self, item: GcodeDialect) -> Gtk.Widget:
        """Creates a DialectRow for the given dialect item."""
        return DialectRow(item)

    def _on_add_clicked(self, button: Gtk.Button):
        """Handles the 'Add New Dialect' button click."""
        parent = cast(Gtk.Window, self.get_ancestor(Gtk.Window))
        # Omit `uid` so the default_factory generates a new one.
        new_dialect = GcodeDialect(
            label=_("New Dialect"),
            description=_("A new custom dialect"),
            laser_on="M4 S{power}",
            laser_off="M5",
            travel_move="G0 X{x} Y{y}",
            linear_move="G1 X{x} Y{y}{f_command}",
            is_custom=True,
            # Fill other fields with safe defaults
            tool_change="",
            set_speed="",
            arc_cw="",
            arc_ccw="",
            air_assist_on="",
            air_assist_off="",
        )
        editor_dialog = DialectEditorDialog(parent, new_dialect)
        editor_dialog.connect(
            "close-request", self._on_new_dialect_dialog_closed, new_dialect
        )
        editor_dialog.present()

    def _on_new_dialect_dialog_closed(
        self, dialog: DialectEditorDialog, new_dialect: GcodeDialect
    ):
        """Adds the new dialect if it was saved."""
        if dialog.saved:
            self.dialect_mgr.add_dialect(dialog.dialect)
