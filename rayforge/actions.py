from typing import TYPE_CHECKING, Dict, Callable, Optional
from gi.repository import Gtk, Gio, GLib  # type: ignore
from .doceditor import layout_actions
from .core.group import Group
from .doceditor.group_cmd import CreateGroupCommand, UngroupCommand

if TYPE_CHECKING:
    from .mainwindow import MainWindow


class ActionManager:
    """Manages the creation and state of all Gio.SimpleActions for the app."""

    def __init__(self, win: "MainWindow"):
        self.win = win
        self.actions: Dict[str, Gio.SimpleAction] = {}

    def register_actions(self):
        """Creates all Gio.SimpleActions and adds them to the window."""
        # Menu & File Actions
        self._add_action("quit", self.win.on_quit_action)
        self._add_action("import", self.win.on_menu_import)
        self._add_action("export", self.win.on_export_clicked)
        self._add_action("about", self.win.show_about_dialog)
        self._add_action("preferences", self.win.show_preferences)
        self._add_action("machine_settings", self.win.show_machine_settings)

        # View Actions
        self._add_stateful_action(
            "show_3d_view",
            self.win.on_show_3d_view,
            GLib.Variant.new_boolean(False),
        )
        self._add_stateful_action(
            "show_workpieces",
            self.win.on_show_workpieces_state_change,
            GLib.Variant.new_boolean(True),  # Default is visible
        )

        # 3D View Control Actions
        self._add_action("view_top", self.win.on_view_top)
        self._add_action("view_front", self.win.on_view_front)
        self._add_action("view_iso", self.win.on_view_iso)
        self._add_stateful_action(
            "view_toggle_perspective",
            self.win.on_view_perspective_state_change,
            GLib.Variant.new_boolean(True),  # Default is perspective
        )

        # Edit & Clipboard Actions
        self._add_action(
            "undo", lambda a, p: self.win.doc.history_manager.undo()
        )
        self._add_action(
            "redo", lambda a, p: self.win.doc.history_manager.redo()
        )
        self._add_action("cut", self.win.on_menu_cut)
        self._add_action("copy", self.win.on_menu_copy)
        self._add_action("paste", self.win.on_paste_requested)
        self._add_action("select_all", self.win.on_select_all)
        self._add_action("duplicate", self.win.on_menu_duplicate)
        self._add_action("remove", self.win.on_menu_remove)
        self._add_action("clear", self.win.on_clear_clicked)

        # Grouping Actions
        self._add_action("group", self.on_group_action)
        self._add_action("ungroup", self.on_ungroup_action)

        # Alignment Actions
        self._add_action(
            "align-h-center",
            lambda a, p: layout_actions.center_horizontally(self.win),
        )
        self._add_action(
            "align-v-center",
            lambda a, p: layout_actions.center_vertically(self.win),
        )
        self._add_action(
            "align-left", lambda a, p: layout_actions.align_left(self.win)
        )
        self._add_action(
            "align-right", lambda a, p: layout_actions.align_right(self.win)
        )
        self._add_action(
            "align-top", lambda a, p: layout_actions.align_top(self.win)
        )
        self._add_action(
            "align-bottom", lambda a, p: layout_actions.align_bottom(self.win)
        )
        self._add_action(
            "spread-h",
            lambda a, p: layout_actions.spread_horizontally(self.win),
        )
        self._add_action(
            "spread-v", lambda a, p: layout_actions.spread_vertically(self.win)
        )
        self._add_action(
            "layout-pixel-perfect",
            lambda a, p: layout_actions.layout_pixel_perfect(self.win),
        )

        # Machine Control Actions
        self._add_action("home", self.win.on_home_clicked)
        self._add_action("frame", self.win.on_frame_clicked)
        self._add_action("send", self.win.on_send_clicked)
        self._add_action("cancel", self.win.on_cancel_clicked)

        # Stateful action for the hold/pause button
        self._add_stateful_action(
            "hold",
            self.win.on_hold_state_change,
            GLib.Variant.new_boolean(False),
        )

    def set_accelerators(self, app: Gtk.Application):
        """Sets keyboard accelerators for the application's actions."""
        app.set_accels_for_action("win.import", ["<Primary>o"])
        app.set_accels_for_action("win.export", ["<Primary>e"])
        app.set_accels_for_action("win.quit", ["<Primary>q"])
        app.set_accels_for_action("win.undo", ["<Primary>z"])
        app.set_accels_for_action(
            "win.redo", ["<Primary>y", "<Primary><Shift>z"]
        )
        app.set_accels_for_action("win.cut", ["<Primary>x"])
        app.set_accels_for_action("win.copy", ["<Primary>c"])
        app.set_accels_for_action("win.paste", ["<Primary>v"])
        app.set_accels_for_action("win.select_all", ["<Primary>a"])
        app.set_accels_for_action("win.duplicate", ["<Primary>d"])
        app.set_accels_for_action("win.remove", ["Delete"])
        app.set_accels_for_action("win.group", ["<Primary>g"])
        app.set_accels_for_action("win.ungroup", ["<Primary><Shift>g"])
        app.set_accels_for_action("win.show_3d_view", ["F12"])
        app.set_accels_for_action("win.view_top", ["1"])
        app.set_accels_for_action("win.view_front", ["2"])
        app.set_accels_for_action("win.view_iso", ["7"])
        app.set_accels_for_action("win.view_toggle_perspective", ["p"])
        app.set_accels_for_action("win.layout-pixel-perfect", ["a"])
        app.set_accels_for_action("win.machine_settings", ["<Primary>less"])
        app.set_accels_for_action("win.preferences", ["<Primary>comma"])
        app.set_accels_for_action("win.about", ["F1"])

    def get_action(self, name: str) -> Gio.SimpleAction:
        """Retrieves a registered action by its name."""
        return self.actions[name]

    def on_group_action(self, action, param):
        """Handler for the 'group' action."""
        selected_elements = self.win.surface.get_selected_elements()
        if len(selected_elements) < 2:
            return

        items_to_group = [elem.data for elem in selected_elements]
        # All items must belong to the same layer to be grouped
        parent_layer = items_to_group[0].parent
        if not all(item.parent is parent_layer for item in items_to_group):
            return  # Should not happen with current selection logic

        cmd = CreateGroupCommand(
            parent_layer, items_to_group, self.win.surface.ops_generator
        )
        self.win.doc.history_manager.execute(cmd)

    def on_ungroup_action(self, action, param):
        """Handler for the 'ungroup' action."""
        selected_elements = self.win.surface.get_selected_elements()

        groups_to_ungroup = [
            elem.data
            for elem in selected_elements
            if isinstance(elem.data, Group)
        ]
        if not groups_to_ungroup:
            return

        cmd = UngroupCommand(groups_to_ungroup, self.win.surface.ops_generator)
        self.win.doc.history_manager.execute(cmd)

    def _add_action(
        self,
        name: str,
        callback: Callable,
        param: Optional[GLib.VariantType] = None,
    ):
        """Helper to create, register, and store a simple Gio.SimpleAction."""
        action = Gio.SimpleAction.new(name, param)
        action.connect("activate", callback)
        self.win.add_action(action)
        self.actions[name] = action

    def _add_stateful_action(
        self, name: str, callback: Callable, initial_state: GLib.Variant
    ):
        """Helper for a stateful action, typically for toggle buttons."""
        action = Gio.SimpleAction.new_stateful(name, None, initial_state)
        # For stateful actions, we ONLY connect to 'change-state'. The default
        # 'activate' handler for boolean actions will correctly call this for
        # us.
        action.connect("change-state", callback)
        self.win.add_action(action)
        self.actions[name] = action
