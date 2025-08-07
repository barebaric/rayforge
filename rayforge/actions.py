from typing import TYPE_CHECKING, Dict, Callable
from gi.repository import Gtk, Gio, GLib  # type: ignore


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
        self._add_action("duplicate", self.win.on_menu_duplicate)
        self._add_action("remove", self.win.on_menu_remove)
        self._add_action("clear", self.win.on_clear_clicked)

        # Alignment Actions
        self._add_action("align-h-center", self.win.on_align_h_center_clicked)
        self._add_action("align-v-center", self.win.on_align_v_center_clicked)
        self._add_action("align-left", self.win.on_align_left_clicked)
        self._add_action("align-right", self.win.on_align_right_clicked)
        self._add_action("align-top", self.win.on_align_top_clicked)
        self._add_action("align-bottom", self.win.on_align_bottom_clicked)
        self._add_action("spread-h", self.win.on_spread_horizontally_clicked)
        self._add_action("spread-v", self.win.on_spread_vertically_clicked)

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
        app.set_accels_for_action("win.duplicate", ["<Primary>d"])
        app.set_accels_for_action("win.remove", ["Delete"])
        app.set_accels_for_action("win.machine_settings", ["<Primary>less"])
        app.set_accels_for_action("win.preferences", ["<Primary>comma"])
        app.set_accels_for_action("win.about", ["F1"])

    def get_action(self, name: str) -> Gio.SimpleAction:
        """Retrieves a registered action by its name."""
        return self.actions[name]

    def _add_action(
        self, name: str, callback: Callable, param: GLib.VariantType = None
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
        # For stateful actions, 'change-state' is for handling requests to
        # change the state.
        action.connect("change-state", callback)
        self.win.add_action(action)
        self.actions[name] = action
