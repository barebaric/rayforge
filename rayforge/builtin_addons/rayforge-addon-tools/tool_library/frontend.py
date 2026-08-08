"""
Frontend for the tool_library addon.

Registers :class:`ToolManagerPage` in the Settings dialog via the
``register_settings_pages`` hook. The page, list widget, and row mirror
the materials/recipes UI conventions
(:class:`~rayforge.ui_gtk.shared.preferences_group.PreferencesGroupWithButton`
list + ``Adw.MessageDialog`` confirmations/edit).
"""

from collections.abc import Callable
from gettext import gettext as _
from typing import Optional, cast

from gi.repository import Adw, Gtk

from rayforge.core.hooks import hookimpl
from rayforge.ui_gtk.icons import get_icon
from rayforge.ui_gtk.settings.registry import SettingsPageRegistry
from rayforge.ui_gtk.shared.gtk import apply_css
from rayforge.ui_gtk.shared.preferences_group import PreferencesGroupWithButton
from rayforge.ui_gtk.shared.preferences_page import TrackedPreferencesPage

from . import get_tool_manager
from .edit_dialog import AddEditToolDialog
from .manager import ToolManager
from .tool import CATEGORY_LABELS, Tool, category_to_name

ADDON_NAME = "tool_library"

apply_css("""
.maturity-warning {
    background-color: alpha(@warning_color, 0.15);
    padding: 10px 28px;
}
""")


class ToolRow(Gtk.Box):
    """A single tool entry in the list."""

    def __init__(
        self,
        tool: Tool,
        on_edit: Callable[[Tool], None],
        on_delete: Callable[[Tool], None],
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.tool = tool
        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        icon = get_icon("tool-change-symbolic")
        icon.set_valign(Gtk.Align.CENTER)
        self.append(icon)

        labels = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, hexpand=True)
        self.append(labels)

        labels.append(
            Gtk.Label(label=tool.name, halign=Gtk.Align.START, xalign=0)
        )
        subtitle = Gtk.Label(
            label=_("{category} \u00b7 \u2300 {diam:g} mm").format(
                category=CATEGORY_LABELS[category_to_name(tool.category)],
                diam=tool.diameter(),
            ),
            halign=Gtk.Align.START,
            xalign=0,
        )
        subtitle.add_css_class("dim-label")
        labels.append(subtitle)

        suffix = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        self.append(suffix)

        edit_btn = Gtk.Button(child=get_icon("edit-symbolic"))
        edit_btn.add_css_class("flat")
        edit_btn.connect("clicked", lambda _w: on_edit(tool))
        suffix.append(edit_btn)

        del_btn = Gtk.Button(child=get_icon("delete-symbolic"))
        del_btn.add_css_class("flat")
        del_btn.connect("clicked", lambda _w: on_delete(tool))
        suffix.append(del_btn)


class ToolListWidget(PreferencesGroupWithButton):
    """Editable list of tools, backed by the :class:`ToolManager`."""

    def __init__(self, manager: ToolManager, **kwargs):
        super().__init__(
            button_label=_("Add Tool"),
            empty_placeholder=_("No tools configured."),
            **kwargs,
        )
        self._mgr = manager
        self._mgr.changed.connect(self._on_changed)
        self._refresh()

    def create_row_widget(self, item: Tool) -> Gtk.Widget:
        return ToolRow(item, self._on_edit, self._on_delete)

    def _on_changed(self, _sender: object) -> None:
        self._refresh()

    def _refresh(self) -> None:
        self.set_items(self._mgr.get_all())

    def _on_add_clicked(self, button: Gtk.Button) -> None:
        self._open_dialog()

    def _on_edit(self, tool: Tool) -> None:
        self._open_dialog(tool)

    def _open_dialog(self, tool: Optional[Tool] = None) -> None:
        root = self.get_root()
        dialog = AddEditToolDialog(
            cast(Gtk.Window, root) if root else None,
            tool=tool,
        )

        def on_response(d, *, response_id):
            if response_id in ("add", "save"):
                self._mgr.save(d.get_tool())
            d.destroy()

        dialog.response.connect(on_response, weak=False)
        dialog.present()

    def _on_delete(self, tool: Tool) -> None:
        root = self.get_root()
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, root) if root else None,
            heading=_("Delete '{name}'?").format(name=tool.name),
            body=_(
                "The tool will be permanently removed. This cannot be undone."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("delete", _("Delete"))
        dialog.set_response_appearance(
            "delete", Adw.ResponseAppearance.DESTRUCTIVE
        )
        dialog.set_default_response("cancel")

        def on_response(d, response_id):
            if response_id == "delete":
                self._mgr.delete(tool.uid)
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()


class ToolManagerPage(TrackedPreferencesPage):
    """Settings page managing the tool library."""

    key = "tools"

    def __init__(self):
        super().__init__(title=_("Tools"), icon_name="tool-change-symbolic")
        warning_group = Adw.PreferencesGroup()
        banner = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            hexpand=True,
        )
        banner.add_css_class("maturity-warning")
        icon = get_icon("warning-symbolic")
        icon.add_css_class("warning")
        label = Gtk.Label(
            label=_(
                "The tool library is experimental and unfinished. "
                "It may not offer backward compatibility."
            ),
            wrap=True,
            xalign=0,
            hexpand=True,
        )
        label.add_css_class("warning-label")
        banner.append(icon)
        banner.append(label)
        warning_group.add(banner)
        self.add(warning_group)
        self.add(
            ToolListWidget(
                get_tool_manager(),
                title=_("Tool Library"),
                description=_("Cutting tools for CNC machining."),
            )
        )


@hookimpl
def register_settings_pages(
    settings_page_registry: SettingsPageRegistry,
) -> None:
    """Contribute the tool-manager page to the Settings dialog."""
    settings_page_registry.register(ToolManagerPage, addon_name=ADDON_NAME)
