from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union, cast

from gi.repository import Adw, Gtk

from ...core.commands import GridCommand
from ...core.entities import Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ...core.constraints import Constraint


class GridTool(SketchTool):
    ICON = "sketch-grid-symbolic"
    LABEL = _("Grid")
    SHORTCUTS = ["gg"]

    def is_available(
        self,
        target: Optional[Union["Point", "Entity", "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        return target is None

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_activate(self):
        self._show_dialog()
        self.element.set_tool("select")

    def _show_dialog(self):
        editor = self.element.editor
        if not editor or not editor.parent_window:
            return

        parent_window = cast(Gtk.Window, editor.parent_window)

        dialog = Adw.MessageDialog(
            transient_for=parent_window,
            modal=True,
            destroy_with_parent=True,
            heading=_("Create Grid"),
        )

        rows_entry = Adw.EntryRow(title=_("Rows"))
        rows_entry.set_input_purpose(Gtk.InputPurpose.NUMBER)
        rows_entry.set_text("3")

        cols_entry = Adw.EntryRow(title=_("Columns"))
        cols_entry.set_input_purpose(Gtk.InputPurpose.NUMBER)
        cols_entry.set_text("3")

        list_box = Gtk.ListBox(
            selection_mode=Gtk.SelectionMode.NONE,
            css_classes=["boxed-list"],
        )
        list_box.append(rows_entry)
        list_box.append(cols_entry)
        dialog.set_extra_child(list_box)

        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("create", _("Create"))
        dialog.set_response_appearance(
            "create", Adw.ResponseAppearance.SUGGESTED
        )
        dialog.set_default_response("create")
        dialog.set_close_response("cancel")

        def on_response(source, response_id):
            if response_id == "create":
                try:
                    rows = int(rows_entry.get_text())
                    cols = int(cols_entry.get_text())
                except ValueError:
                    return

                if rows < 2 or cols < 2:
                    return

                cmd = GridCommand(
                    self.element.sketch,
                    rows=rows,
                    cols=cols,
                )
                self.element.execute_command(cmd)

            dialog.close()

        dialog.connect("response", on_response)
        dialog.present()
