"""A dialog that asks for a new name for a document item."""

from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...core.item import DocItem


class RenameDialog(Adw.MessageDialog):
    """Asks for a new name and applies it via an undoable command."""

    def __init__(self, parent: Gtk.Window | None, item: DocItem, rename):
        super().__init__(transient_for=parent)
        self._item = item
        self._rename = rename

        self.set_heading(_("Rename"))
        self.set_body(item.name or "")

        self._entry = Gtk.Entry()
        self._entry.set_text(item.name or "")
        self._entry.set_activates_default(True)
        self._entry.set_margin_top(12)
        self.set_extra_child(self._entry)

        self.add_response("cancel", _("_Cancel"))
        self.add_response("rename", _("_Rename"))
        self.set_default_response("rename")
        self.set_close_response("cancel")
        self.set_response_appearance(
            "rename", Adw.ResponseAppearance.SUGGESTED
        )
        self.connect("response", self._on_response)

    def _on_response(self, dialog, response):
        if response == "rename":
            self._rename(self._item, self._entry.get_text())
