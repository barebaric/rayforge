from gettext import gettext as _
from typing import Callable, Optional
from gi.repository import Adw, GLib, Gtk
from ..shared.gtk import apply_css
from ...machine.models.dialect import GcodeDialect, BUILTIN_DIALECTS


css = """
.dialect-template-list {
    background: none;
}
"""


class DialectTemplateSelectorDialog(Adw.MessageDialog):
    """
    A dialog for selecting a dialect template from the built-in dialects.

    The dialog is confirmed by activating a row (double-click or Enter).
    """

    class _TemplateRow(Adw.ActionRow):
        """A custom row to hold a reference to its dialect template."""

        def __init__(self, template: GcodeDialect, **kwargs):
            super().__init__(**kwargs)
            self.template: GcodeDialect = template

    def __init__(
        self,
        on_selected: Optional[Callable[[GcodeDialect], None]] = None,
        title: Optional[str] = None,
        body: Optional[str] = None,
        **kwargs,
    ):
        """Initializes the Dialect Template Selector dialog.

        Args:
            on_selected: Callback called when a template is selected.
            title: Dialog heading text.
            body: Dialog body text.
        """
        super().__init__(**kwargs)
        self._on_selected = on_selected
        self.set_heading(title or _("Select a Template"))
        self.set_body(
            body or _("Choose a built-in dialect as a starting point.")
        )
        self.set_transient_for(kwargs.get("transient_for"))

        apply_css(css)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.set_margin_top(12)
        content.set_size_request(460, 400)

        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        scrolled_window.set_min_content_height(250)
        scrolled_window.set_vexpand(True)
        scrolled_window.add_css_class("card")
        content.append(scrolled_window)

        self.template_list_box = Gtk.ListBox()
        self.template_list_box.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.template_list_box.add_css_class("dialect-template-list")
        self.template_list_box.connect("row-activated", self._on_row_activated)
        scrolled_window.set_child(self.template_list_box)

        self._populate_template_list()

        self.set_extra_child(content)

        self.add_response("cancel", _("Cancel"))
        self.set_default_response("cancel")

    def _populate_template_list(self):
        """Fills the list box with available built-in dialect templates."""
        sorted_templates = sorted(
            BUILTIN_DIALECTS, key=lambda t: t.label.lower()
        )

        for template in sorted_templates:
            subtitle = GLib.markup_escape_text(template.description, -1)
            row = self._TemplateRow(
                template=template,
                title=template.label,
                subtitle=subtitle,
                activatable=True,
            )
            self.template_list_box.append(row)

    def _on_row_activated(self, listbox: Gtk.ListBox, row: _TemplateRow):
        """Handles row activation, calls callback, and closes the dialog."""
        if self._on_selected:
            self._on_selected(row.template)
        self.close()
