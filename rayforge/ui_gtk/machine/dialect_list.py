from typing import cast
from gettext import gettext as _
from gi.repository import Gtk, Adw
from ...machine.models.dialect import GcodeDialect, get_available_dialects
from ...context import get_context
from ..icons import get_icon
from ..shared.preferences_group import PreferencesGroupWithButton
from .dialect_editor import DialectEditorDialog
from .template_selector import DialectTemplateSelectorDialog


class DialectRow(Gtk.Box):
    """A widget representing a single Dialect in a ListBox."""

    def __init__(self, dialect: GcodeDialect, machine=None):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.dialect = dialect
        self.dialect_mgr = get_context().dialect_mgr
        self.machine = machine
        self._setup_ui()

    def _setup_ui(self):
        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        info_box.set_hexpand(True)
        info_box.set_valign(Gtk.Align.CENTER)
        self.append(info_box)

        title_label = Gtk.Label(
            label=self.dialect.label,
            halign=Gtk.Align.START,
            xalign=0,
        )
        info_box.append(title_label)

        if self.dialect.description:
            desc_label = Gtk.Label(
                label=self.dialect.description,
                halign=Gtk.Align.START,
                xalign=0,
                wrap=True,
            )
            desc_label.add_css_class("dim-label")
            desc_label.add_css_class("caption")
            info_box.append(desc_label)

        suffix_box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        self.append(suffix_box)

        edit_button = Gtk.Button(child=get_icon("edit-symbolic"))
        edit_button.add_css_class("flat")
        edit_button.connect("clicked", self._on_edit_clicked)
        suffix_box.append(edit_button)

        delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        delete_button.add_css_class("flat")
        delete_button.connect("clicked", self._on_delete_clicked)
        suffix_box.append(delete_button)

        self.select_button = Gtk.ToggleButton()
        self.select_button.add_css_class("flat")
        self.select_button.set_child(get_icon("check-symbolic"))
        self.select_button.set_tooltip_text(_("Select this dialect"))
        self.select_button.connect("toggled", self._on_select_toggled)
        self.select_button.set_valign(Gtk.Align.CENTER)
        suffix_box.append(self.select_button)

        self._update_selection_state()

    def _update_selection_state(self):
        if self.machine:
            is_selected = self.machine.dialect_uid == self.dialect.uid
            self.select_button.set_active(is_selected)

    def _on_select_toggled(self, button: Gtk.ToggleButton):
        if button.get_active():
            if self.machine and self.machine.dialect_uid != self.dialect.uid:
                self.machine.set_dialect_uid(self.dialect.uid)
        else:
            # Prevent deselecting - always keep one dialect selected
            button.set_active(True)

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
            machines = get_context().machine_mgr.get_machines()
            machines_using = self.dialect_mgr.get_machines_using_dialect(
                self.dialect, machines
            )
            if machines_using:
                machine_names = ", ".join(m.name for m in machines_using)
                parent = cast(Gtk.Window, self.get_ancestor(Gtk.Window))
                error_dialog = Adw.MessageDialog(
                    transient_for=parent,
                    heading=_("Cannot Delete Dialect"),
                    body=_(
                        "This dialect is still used by the following "
                        "machine(s): {machines}"
                    ).format(machines=machine_names),
                )
                error_dialog.add_response("ok", _("OK"))
                error_dialog.set_default_response("ok")
                error_dialog.present()
                return
            self.dialect_mgr.delete_dialect(self.dialect, machines)


class DialectListEditor(PreferencesGroupWithButton):
    """An Adwaita widget for managing a list of G-code dialects."""

    def __init__(self, machine=None, **kwargs):
        super().__init__(button_label=_("Create from Template"), **kwargs)
        self.machine = machine
        self.dialect_mgr = get_context().dialect_mgr
        self._row_widgets: list[DialectRow] = []
        self._setup_ui()
        self.dialect_mgr.dialects_changed.connect(self._on_dialects_changed)
        if self.machine:
            self.machine.changed.connect(self._on_machine_changed)
        self.connect("destroy", self._on_destroy)
        self._on_dialects_changed()

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
        if self.machine:
            self.machine.changed.disconnect(self._on_machine_changed)

    def _on_machine_changed(self, sender, **kwargs):
        """Update selection state when machine changes."""
        for row in self._row_widgets:
            row._update_selection_state()

    def _on_dialects_changed(self, sender=None, **kwargs):
        """Callback to rebuild the list when the dialect manager signals."""
        self._row_widgets.clear()
        all_dialects = get_available_dialects()
        custom_dialects = [d for d in all_dialects if d.is_custom]
        sorted_dialects = sorted(custom_dialects, key=lambda d: d.label)
        self.set_items(sorted_dialects)

    def create_row_widget(self, item: GcodeDialect) -> Gtk.Widget:
        """Creates a DialectRow for the given dialect item."""
        row = DialectRow(item, self.machine)
        self._row_widgets.append(row)
        return row

    def _on_add_clicked(self, button: Gtk.Button):
        """Handles the 'Create from Template' button click."""
        parent = cast(Gtk.Window, self.get_ancestor(Gtk.Window))
        dialog = DialectTemplateSelectorDialog(
            transient_for=parent,
            on_selected=self._on_template_selected,
        )
        dialog.present()

    def _on_template_selected(self, template: GcodeDialect):
        """Called when a template is selected from the dialog."""
        parent = cast(Gtk.Window, self.get_ancestor(Gtk.Window))
        new_label = _("{label} (Copy)").format(label=template.label)
        new_dialect = template.copy_as_custom(new_label=new_label)

        editor_dialog = DialectEditorDialog(parent, new_dialect)
        editor_dialog.connect(
            "close-request", self._on_new_dialect_dialog_closed
        )
        editor_dialog.present()

    def _on_new_dialect_dialog_closed(self, dialog: DialectEditorDialog):
        """Adds the new dialect if it was saved."""
        if dialog.saved:
            self.dialect_mgr.add_dialect(dialog.dialect)
