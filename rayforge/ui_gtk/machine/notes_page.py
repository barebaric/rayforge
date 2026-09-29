"""Machine-specific and profile-maintained notes."""

from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...machine.models.machine import Machine
from ..icons import get_icon
from ..shared.markdown import MarkdownEditorDialog, MarkdownView
from ..shared.preferences_page import TrackedPreferencesPage


class NotesPage(TrackedPreferencesPage):
    key = "notes"
    path_prefix = "/machine-settings/"

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(
            title=_("Notes"),
            icon_name="help-about-symbolic",
            **kwargs,
        )
        self.machine = machine

        user_group = Adw.PreferencesGroup(title=_("My Notes"))
        self.add(user_group)

        self.user_notes_view = MarkdownView(machine.user_notes)
        user_group.add(self.user_notes_view)

        self.empty_label = Gtk.Label(
            label=_("Add your own notes for this machine"),
            xalign=0,
        )
        self.empty_label.add_css_class("dim-label")
        user_group.add(self.empty_label)

        self.edit_button = Gtk.Button(
            halign=Gtk.Align.START,
            margin_top=12,
            margin_bottom=12,
        )
        button_content = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            halign=Gtk.Align.CENTER,
        )
        button_content.append(get_icon("edit-symbolic"))
        button_content.append(Gtk.Label(label=_("Edit My Notes")))
        self.edit_button.set_child(button_content)
        self.edit_button.connect("clicked", self._on_edit_clicked)
        user_group.add(self.edit_button)

        self.device_group = Adw.PreferencesGroup(title=_("Device Notes"))
        self.add(self.device_group)
        self.device_notes_view = MarkdownView(machine.device_notes or "")
        self.device_group.add(self.device_notes_view)
        self.device_group.set_visible(
            bool(machine.device_notes and machine.device_notes.strip())
        )

        self._editor_dialog: MarkdownEditorDialog | None = None
        self._sync_user_notes()

    def _sync_user_notes(self):
        notes = self.machine.user_notes
        self.user_notes_view.set_text(notes)
        has_notes = bool(notes.strip())
        self.user_notes_view.set_visible(has_notes)
        self.empty_label.set_visible(not has_notes)

    def _on_edit_clicked(self, _button):
        if self._editor_dialog is not None:
            if self._editor_dialog.get_visible():
                self._editor_dialog.present()
                return
            self._editor_dialog.destroy()
        self._editor_dialog = MarkdownEditorDialog(
            title=_("Edit My Notes"),
            description=_(
                "These notes are stored with this machine and are never "
                "overwritten by device profile updates."
            ),
            initial_text=self.machine.user_notes,
            transient_for=self.get_ancestor(Gtk.Window),
        )
        # The editor outlives this page when the settings window closes,
        # so it must keep its save target alive.
        self._editor_dialog.saved.connect(self._on_notes_saved, weak=False)
        self._editor_dialog.present()

    def _on_notes_saved(self, _dialog, text: str):
        self.machine.user_notes = text
        self.machine.changed.send(self.machine)
        self._sync_user_notes()


__all__ = ["NotesPage"]
