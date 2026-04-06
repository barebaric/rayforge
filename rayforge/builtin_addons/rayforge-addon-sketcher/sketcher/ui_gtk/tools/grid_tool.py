from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union, cast

from gi.repository import Adw, Gtk

from rayforge.ui_gtk.shared.adwfix import get_spinrow_int
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

        rows_adj = Gtk.Adjustment(lower=2, upper=100, step_increment=1)
        rows_row = Adw.SpinRow(
            title=_("Rows"),
            adjustment=rows_adj,
            digits=0,
            value=3,
        )

        cols_adj = Gtk.Adjustment(lower=2, upper=100, step_increment=1)
        cols_row = Adw.SpinRow(
            title=_("Columns"),
            adjustment=cols_adj,
            digits=0,
            value=3,
        )

        list_box = Gtk.ListBox(
            selection_mode=Gtk.SelectionMode.NONE,
            css_classes=["boxed-list"],
        )
        list_box.append(rows_row)
        list_box.append(cols_row)
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
                rows = get_spinrow_int(rows_row)
                cols = get_spinrow_int(cols_row)

                cmd = GridCommand(
                    self.element.sketch,
                    rows=rows,
                    cols=cols,
                )
                self.element.execute_command(cmd)

            dialog.close()

        dialog.connect("response", on_response)
        dialog.present()
