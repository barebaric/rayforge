from typing import TYPE_CHECKING, Dict, Callable, Optional, cast
from gi.repository import Gtk, Gio, GLib
from .core.group import Group
from .core.item import DocItem
from .core.layer import Layer
from .core.workpiece import WorkPiece
from .doceditor.ui.add_tabs_popover import AddTabsPopover


if TYPE_CHECKING:
    from .mainwindow import MainWindow


class ActionManager:
    """Manages the creation and state of all Gio.SimpleActions for the app."""

    def __init__(self, win: "MainWindow"):
        self.win = win
        self.actions: Dict[str, Gio.SimpleAction] = {}
        # A convenient alias to the central controller
        self.editor = self.win.doc_editor
        self.doc = self.editor.doc

        # Connect to doc signals to update action states
        self.doc.descendant_added.connect(self.update_action_states)
        self.doc.descendant_removed.connect(self.update_action_states)
        self.win.surface.selection_changed.connect(self.update_action_states)
        self.win.surface.context_changed.connect(self.update_action_states)

    def register_actions(self):
        """Creates all Gio.SimpleActions and adds them to the window."""
        # Menu & File Actions
        self._add_action("quit", self.win.on_quit_action)
        self._add_action("import", self.win.on_menu_import)
        self._add_action("export", self.win.on_export_clicked)
        self._add_action("about", self.win.show_about_dialog)
        self._add_action("preferences", self.win.show_preferences)
        self._add_action("machine-settings", self.win.show_machine_settings)

        # Tools Actions
        self._add_action("material_test", self.win.on_show_material_test)

        # View Actions
        self._add_stateful_action(
            "show_3d_view",
            self.win.on_show_3d_view,
            GLib.Variant.new_boolean(False),
        )
        self._add_stateful_action(
            "show_workpieces",
            self.win.on_show_workpieces_state_change,
            GLib.Variant.new_boolean(True),
        )
        self._add_stateful_action(
            "toggle_camera_view",
            self.win.on_toggle_camera_view_state_change,
            GLib.Variant.new_boolean(True),
        )
        self._add_stateful_action(
            "toggle_travel_view",
            self.win.on_toggle_travel_view_state_change,
            GLib.Variant.new_boolean(False),
        )
        self._add_stateful_action(
            "toggle_gcode_preview",
            self.win.on_toggle_gcode_preview_state_change,
            GLib.Variant.new_boolean(False),
        )

        self._add_stateful_action(
            "simulate_mode",
            self.win.simulator_cmd.toggle_mode,
            GLib.Variant.new_boolean(False),
        )

        # 3D View Control Actions
        self._add_action("view_top", self.win.on_view_top)
        self._add_action("view_front", self.win.on_view_front)
        self._add_action("view_iso", self.win.on_view_iso)
        self._add_stateful_action(
            "view_toggle_perspective",
            self.win.on_view_perspective_state_change,
            GLib.Variant.new_boolean(True),
        )

        # Edit & Clipboard Actions
        self._add_action(
            "undo", lambda a, p: self.editor.history_manager.undo()
        )
        # Primary redo action, linked to menu and toolbar
        self._add_action(
            "redo", lambda a, p: self.editor.history_manager.redo()
        )
        # Secondary, hidden redo action for the alternate shortcut
        self._add_action(
            "redo_alt", lambda a, p: self.editor.history_manager.redo()
        )
        self._add_action("cut", self.win.on_menu_cut)
        self._add_action("copy", self.win.on_menu_copy)
        self._add_action("paste", self.win.on_paste_requested)
        self._add_action("select_all", self.win.on_select_all)
        self._add_action("duplicate", self.win.on_menu_duplicate)
        self._add_action("remove", self.win.on_menu_remove)
        self._add_action("clear", self.win.on_clear_clicked)

        # Item Actions
        self._add_action("new_sketch", self.win.on_new_sketch)
        self._add_action("edit_sketch", self.win.on_edit_sketch)
        self._add_action("export_sketch", self.win.on_export_sketch)

        # Layer Management Actions
        self._add_action("layer-move-up", self.on_layer_move_up)
        self._add_action("layer-move-down", self.on_layer_move_down)
        self._add_action("add_stock", self.on_add_stock)

        # Grouping Actions
        self._add_action("group", self.on_group_action)
        self._add_action("ungroup", self.on_ungroup_action)

        # Split Action
        self._add_action("split", self.on_split_action)

        # Tabbing Actions
        self._add_action("add-tabs-equidistant", self.on_add_tabs_equidistant)
        self._add_action("add-tabs-cardinal", self.on_add_tabs_cardinal)
        self._add_action("tab-add", self.on_tab_add)
        self._add_action("tab-remove", self.on_tab_remove)
        self._add_stateful_action(
            "show_tabs",
            self.on_show_tabs_state_change,
            GLib.Variant.new_boolean(True),
        )

        # Alignment Actions
        self._add_action("align-h-center", self.on_align_h_center)
        self._add_action("align-v-center", self.on_align_v_center)
        self._add_action("align-left", self.on_align_left)
        self._add_action("align-right", self.on_align_right)
        self._add_action("align-top", self.on_align_top)
        self._add_action("align-bottom", self.on_align_bottom)
        self._add_action("spread-h", self.on_spread_h)
        self._add_action("spread-v", self.on_spread_v)
        self._add_action("layout-pixel-perfect", self.on_layout_pixel_perfect)

        # Transform Actions
        self._add_action("flip-horizontal", self.on_flip_horizontal)
        self._add_action("flip-vertical", self.on_flip_vertical)

        # Macro Actions
        self._add_action(
            "execute-macro",
            self.win.on_execute_macro,
            GLib.VariantType.new("s"),
        )

        # Machine Control Actions
        self._add_action("machine-home", self.win.on_home_clicked)
        self._add_action("machine-frame", self.win.on_frame_clicked)
        self._add_action("machine-send", self.win.on_send_clicked)
        self._add_action("machine-cancel", self.win.on_cancel_clicked)
        self._add_action(
            "machine-clear-alarm", self.win.on_clear_alarm_clicked
        )
        self._add_action("machine-jog", self.win.on_jog_clicked)

        # Stateful action for the hold/pause button
        self._add_stateful_action(
            "machine-hold",
            self.win.on_hold_state_change,
            GLib.Variant.new_boolean(False),
        )

        self._add_stateful_action(
            "toggle-focus",
            self.win.on_toggle_focus_state_change,
            GLib.Variant.new_boolean(False),
        )

        self.update_action_states()

    def update_action_states(self, *args, **kwargs):
        """Updates the enabled state of actions based on document state."""
        self.actions["add_stock"].set_enabled(True)
        self.actions["new_sketch"].set_enabled(True)

        target_workpieces = self._get_workpieces_for_tabbing()
        can_add_tabs = any(wp.boundaries for wp in target_workpieces)
        self.actions["add-tabs-equidistant"].set_enabled(can_add_tabs)
        self.actions["add-tabs-cardinal"].set_enabled(can_add_tabs)

        # Update context-sensitive tab actions based on the surface's public
        # state
        context = self.win.surface.right_click_context
        can_add_single_tab = context and context.get("type") == "geometry"
        can_remove_single_tab = context and context.get("type") == "tab"
        self.actions["tab-add"].set_enabled(bool(can_add_single_tab))
        self.actions["tab-remove"].set_enabled(bool(can_remove_single_tab))

        # Update layout-pixel-perfect action state
        current_layer = self.doc.active_layer
        has_workpieces = (
            current_layer and len(current_layer.get_descendants(WorkPiece)) > 0
        )
        self.actions["layout-pixel-perfect"].set_enabled(has_workpieces)

        # Update split action state
        selected_wps = self.win.surface.get_selected_workpieces()
        self.actions["split"].set_enabled(bool(selected_wps))

        # Update edit_sketch and export_sketch action states
        # Only enable if exactly one workpiece is selected and it is a sketch
        can_edit_or_export_sketch = False
        if len(selected_wps) == 1:
            wp = selected_wps[0]
            # Is it a sketch-based WorkPiece?
            if wp.sketch_uid:
                can_edit_or_export_sketch = True

        if "export_sketch" in self.actions:
            self.actions["export_sketch"].set_enabled(
                can_edit_or_export_sketch
            )
        if "edit_sketch" in self.actions:
            self.actions["edit_sketch"].set_enabled(can_edit_or_export_sketch)

    def on_add_stock(self, action, param):
        """Handler for the 'add_stock' action."""
        self.editor.stock.add_stock()

    def _get_workpieces_for_tabbing(self) -> list[WorkPiece]:
        """
        Helper to get a list of workpieces to apply tabs to, using the
        surface's selection API.
        """
        # Use the dedicated surface method to get selected workpieces. This
        # correctly handles nested items inside groups.
        selected_workpieces = self.win.surface.get_selected_workpieces()

        if not selected_workpieces:
            # If the selection is empty or contains no workpieces, fall back to
            # processing all workpieces in the entire document.
            return list(self.doc.get_descendants(WorkPiece))
        else:
            # Otherwise, return the unique list derived from the selection.
            return selected_workpieces

    def on_add_tabs_equidistant(self, action, param):
        """Opens the popover for adding equidistant tabs."""
        workpieces_to_process = self._get_workpieces_for_tabbing()
        valid_workpieces = [
            wp
            for wp in workpieces_to_process
            if wp.boundaries
            and wp.layer
            and wp.layer.workflow
            and wp.layer.workflow.has_steps()
        ]

        if not valid_workpieces:
            return

        # The popover needs to be parented to the SplitMenuButton's main button
        button = self.win.toolbar.tab_menu_button.main_button
        popover = AddTabsPopover(
            editor=self.editor, workpieces=valid_workpieces
        )
        popover.set_parent(button)
        popover.popup()

    def on_add_tabs_cardinal(self, action, param):
        """Handler for adding cardinal tabs to a workpiece."""
        workpieces_to_process = self._get_workpieces_for_tabbing()
        if not workpieces_to_process:
            return

        # 1. Execute the command to update the data model.
        for workpiece in workpieces_to_process:
            if not (
                workpiece.layer
                and workpiece.layer.workflow
                and workpiece.layer.workflow.has_steps()
            ):
                continue

            self.editor.tab.add_cardinal_tabs(
                workpiece=workpiece,
                width=2.0,
            )

        # 2. Ensure the UI state is visible.
        show_tabs_action = self.get_action("show_tabs")
        state = show_tabs_action.get_state()
        if not (state and state.get_boolean()):
            show_tabs_action.set_state(GLib.Variant.new_boolean(True))

    def on_tab_add(self, action, param):
        """Handler for adding a single tab via context menu."""
        context = self.win.surface.right_click_context
        if context and context.get("type") == "geometry":
            self.editor.add_tab_from_context(context)

    def on_tab_remove(self, action, param):
        """Handler for removing a single tab via context menu."""
        context = self.win.surface.right_click_context
        if context and context.get("type") == "tab":
            self.editor.remove_tab_from_context(context)

    def on_show_tabs_state_change(self, action, state):
        """
        Handler for the global tab visibility state change. This is the
        controller that receives the user's intent to change the state.
        """
        is_visible = state.get_boolean()
        # 1. Perform the action by telling the surface to update its state.
        self.win.surface.set_global_tab_visibility(is_visible)
        # 2. Confirm the state change, which updates the action and widgets.
        action.set_state(state)

    def register_shortcuts(self, controller: Gtk.ShortcutController):
        """
        Populates the given ShortcutController with all application shortcuts.
        """
        shortcuts = {
            # File
            "win.import": "<Primary>i",
            "win.export": "<Primary>e",
            "win.quit": "<Primary>q",
            # Edit
            "win.undo": "<Primary>z",
            "win.redo": "<Primary>y",
            "win.redo_alt": "<Primary><Shift>z",
            "win.cut": "<Primary>x",
            "win.copy": "<Primary>c",
            "win.paste": "<Primary>v",
            "win.select_all": "<Primary>a",
            "win.duplicate": "<Primary>d",
            "win.remove": "Delete",
            "win.clear": "<Primary><Shift>Delete",
            "win.preferences": "<Primary>comma",
            # View
            "win.show_workpieces": "h",
            "win.show_tabs": "t",
            "win.toggle_camera_view": "<Alt>c",
            "win.toggle_travel_view": "<Primary><Shift>t",
            "win.toggle_gcode_preview": "<Primary><Shift>g",
            "win.show_3d_view": "F12",
            "win.simulate_mode": "<Primary><Shift>s",
            "win.view_top": "1",
            "win.view_front": "2",
            "win.view_iso": "7",
            "win.view_toggle_perspective": "p",
            # Object
            "win.add_stock": "<Alt>s",
            "win.new_sketch": "<Primary>n",
            "win.add-tabs-equidistant": "<Alt>t",
            # Arrange
            "win.group": "<Primary>g",
            "win.ungroup": "<Primary>u",
            "win.split": "<Alt>w",
            "win.layer-move-up": "<Primary>Page_Up",
            "win.layer-move-down": "<Primary>Page_Down",
            "win.align-left": "<Primary><Shift>Left",
            "win.align-right": "<Primary><Shift>Right",
            "win.align-top": "<Primary><Shift>Up",
            "win.align-bottom": "<Primary><Shift>Down",
            "win.align-h-center": "<Primary><Shift>Home",
            "win.align-v-center": "<Primary><Shift>End",
            "win.spread-h": "<Primary><Shift>h",
            "win.spread-v": "<Primary><Shift>v",
            "win.layout-pixel-perfect": "<Alt>a",
            "win.flip-horizontal": "<Shift>h",
            "win.flip-vertical": "<Shift>v",
            # Machine & Help
            "win.machine-settings": "<Primary>less",
            "win.machine-jog": "<Primary>j",
            "win.about": "F1",
        }

        for action_name, shortcut_str in shortcuts.items():
            shortcut = Gtk.Shortcut.new(
                Gtk.ShortcutTrigger.parse_string(shortcut_str),
                Gtk.NamedAction.new(action_name),
            )
            controller.add_shortcut(shortcut)

    def get_action(self, name: str) -> Gio.SimpleAction:
        """Retrieves a registered action by its name."""
        return self.actions[name]

    def on_layer_move_up(self, action, param):
        """Handler for the 'layer-move-up' action."""
        self.editor.layer.move_selected_to_adjacent_layer(
            self.win.surface, direction=-1
        )

    def on_layer_move_down(self, action, param):
        """Handler for the 'layer-move-down' action."""
        self.editor.layer.move_selected_to_adjacent_layer(
            self.win.surface, direction=1
        )

    def on_group_action(self, action, param):
        """Handler for the 'group' action."""
        selected_elements = self.win.surface.get_selected_elements()
        if len(selected_elements) < 2:
            return

        items_to_group = [
            elem.data
            for elem in selected_elements
            if isinstance(elem.data, DocItem)
        ]
        # All items must belong to the same layer to be grouped
        parent_layer = cast(Layer, items_to_group[0].parent)
        if not parent_layer or not all(
            item.parent is parent_layer for item in items_to_group
        ):
            return

        new_group = self.editor.group.group_items(parent_layer, items_to_group)
        if new_group:
            self.win.surface.select_items([new_group])

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

        self.editor.group.ungroup_items(groups_to_ungroup)
        # The selection will be automatically updated by the history changed
        # signal handler.

    def on_split_action(self, action, param):
        """Handler for the 'split' action."""
        selected_workpieces = self.win.surface.get_selected_workpieces()
        if not selected_workpieces:
            return

        new_items = self.editor.split.split_items(selected_workpieces)
        if new_items:
            self.win.surface.select_items(new_items)

    # --- Alignment Action Handlers ---

    def on_align_h_center(self, action, param):
        items = list(self.win.surface.get_selected_items())
        w, _ = self.win.surface.get_size_mm()
        self.editor.layout.center_horizontally(items, w)

    def on_align_v_center(self, action, param):
        items = list(self.win.surface.get_selected_items())
        _, h = self.win.surface.get_size_mm()
        self.editor.layout.center_vertically(items, h)

    def on_align_left(self, action, param):
        items = list(self.win.surface.get_selected_items())
        self.editor.layout.align_left(items)

    def on_align_right(self, action, param):
        items = list(self.win.surface.get_selected_items())
        w, _ = self.win.surface.get_size_mm()
        self.editor.layout.align_right(items, w)

    def on_align_top(self, action, param):
        items = list(self.win.surface.get_selected_items())
        _, h = self.win.surface.get_size_mm()
        self.editor.layout.align_top(items, h)

    def on_align_bottom(self, action, param):
        items = list(self.win.surface.get_selected_items())
        self.editor.layout.align_bottom(items)

    def on_spread_h(self, action, param):
        items = list(self.win.surface.get_selected_items())
        self.editor.layout.spread_horizontally(items)

    def on_spread_v(self, action, param):
        items = list(self.win.surface.get_selected_items())
        self.editor.layout.spread_vertically(items)

    def on_layout_pixel_perfect(self, action, param):
        items = list(self.win.surface.get_selected_items())
        self.editor.layout.layout_pixel_perfect(items)

    def on_flip_horizontal(self, action, param):
        """Handler for the 'flip-horizontal' action."""
        items = list(self.win.surface.get_selected_items())
        self.editor.transform.flip_horizontal(items)

    def on_flip_vertical(self, action, param):
        """Handler for the 'flip-vertical' action."""
        items = list(self.win.surface.get_selected_items())
        self.editor.transform.flip_vertical(items)

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
