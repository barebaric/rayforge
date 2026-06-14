from gettext import gettext as _
from typing import Callable, Optional

from gi.repository import Adw, Gtk

from ...machine.device.lightburn_importer import ImportSummary


class LBDevImportDialog(Adw.MessageDialog):
    """Modal dialog warning about incomplete LightBurn imports.

    Displays a warning that the imported profile may be incomplete,
    followed by a summary table of the values that were mapped.
    The user may proceed with the import or cancel.
    """

    def __init__(
        self,
        parent: Gtk.Window,
        summary: ImportSummary,
        on_import: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(
            transient_for=parent,
            modal=True,
            heading=_("Import LightBurn profile?"),
            body=_(
                "LightBurn device profiles contain only basic machine "
                "settings. The imported profile may be incomplete. "
                "After import, please review and configure any "
                "additional settings such as laser heads, homing, "
                "end stops, G-code dialect, macros, and rotary "
                "modules."
            ),
            **kwargs,
        )
        self._on_import = on_import

        self.set_size_request(500, -1)
        self._build_extra_child(summary)

        self.add_response("cancel", _("Cancel"))
        self.set_default_response("cancel")
        self.set_close_response("cancel")
        self.add_response("import", _("Import Anyway"))
        self.set_response_appearance(
            "import", Adw.ResponseAppearance.SUGGESTED
        )
        self.connect("response", self._on_response)

    def _build_extra_child(self, summary: ImportSummary):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        outer.set_margin_top(12)

        heading = Gtk.Label(
            label=_("The following values will be imported:"),
            xalign=0.0,
        )
        heading.add_css_class("caption-heading")
        outer.append(heading)

        list_box = Gtk.ListBox(css_classes=["boxed-list"])
        list_box.set_selection_mode(Gtk.SelectionMode.NONE)

        for field, value in summary.to_items():
            row = Adw.ActionRow(title=field, subtitle=value)
            list_box.append(row)

        outer.append(list_box)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_propagate_natural_height(True)
        scrolled.set_max_content_height(300)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_child(outer)

        self.set_extra_child(scrolled)

    def _on_response(self, dialog, response_id):
        self.destroy()
        if response_id == "import" and self._on_import:
            self._on_import()
