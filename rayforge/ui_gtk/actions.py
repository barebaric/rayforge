import logging
from typing import TYPE_CHECKING, Dict, Callable, List, Optional, cast
from gi.repository import Gtk, Gio, GLib
from gettext import gettext as _
from ..context import get_context
from ..core.group import Group
from ..core.item import DocItem
from ..core.layer import Layer
from ..core.stock import StockItem
from ..core.workpiece import WorkPiece
from ..doceditor.layout.registry import layout_registry
from .action_registry import action_registry, MenuPlacement, ToolbarPlacement
from .doceditor.add_tabs_popover import AddTabsPopover
from .doceditor.stock_properties_dialog import StockPropertiesDialog
from .shared.keyboard import PRIMARY_ACCEL


if TYPE_CHECKING:
    from .mainwindow import MainWindow

logger = logging.getLogger(__name__)

SHORTCUTS = {
    # File
    "win.new": f"{PRIMARY_ACCEL}n",
    "win.open": f"{PRIMARY_ACCEL}o",
    "win.save": f"{PRIMARY_ACCEL}s",
    "win.save-as": f"{PRIMARY_ACCEL}<Shift>s",
    "win.import": f"{PRIMARY_ACCEL}i",
    "win.export": f"{PRIMARY_ACCEL}e",
    "win.quit": f"{PRIMARY_ACCEL}q",
    # Edit
    "win.undo": f"{PRIMARY_ACCEL}z",
    "win.redo": f"{PRIMARY_ACCEL}y",
    "win.redo_alt": f"{PRIMARY_ACCEL}<Shift>z",
    "win.cut": f"{PRIMARY_ACCEL}x",
    "win.copy": f"{PRIMARY_ACCEL}c",
    "win.paste": f"{PRIMARY_ACCEL}v",
    "win.select_all": f"{PRIMARY_ACCEL}a",
    "win.duplicate": f"{PRIMARY_ACCEL}d",
    "win.remove": "Delete",
    "win.clear": f"{PRIMARY_ACCEL}<Shift>Delete",
    "win.settings": f"{PRIMARY_ACCEL}comma",
    # View
    "win.show_workpieces": "h",
    "win.show_tabs": "t",
    "win.toggle_camera_view": "<Ctrl><Alt>c",
    "win.toggle_bottom_panel": f"{PRIMARY_ACCEL}l",
    "win.toggle_travel_view": f"{PRIMARY_ACCEL}<Shift>t",
    "win.show_3d_view": "F12",
    "win.recalculate": "F5",
    "win.force-recalculate": "<Shift>F5",
    "win.view_top": "1",
    "win.view_front": "2",
    "win.view_right": "3",
    "win.view_left": "4",
    "win.view_back": "5",
    "win.view_iso": "7",
    "win.view_toggle_perspective": "p",
    # Object
    "win.add_stock": "<Ctrl><Alt>s",
    "win.add-tabs-equidistant": "<Ctrl><Alt>t",
    # Arrange
    "win.group": f"{PRIMARY_ACCEL}g",
    "win.ungroup": f"{PRIMARY_ACCEL}u",
    "win.split": "<Ctrl><Alt>w",
    "win.layer-move-up": f"{PRIMARY_ACCEL}Page_Up",
    "win.layer-move-down": f"{PRIMARY_ACCEL}Page_Down",
    "win.align-left": f"{PRIMARY_ACCEL}<Shift>Left",
    "win.align-right": f"{PRIMARY_ACCEL}<Shift>Right",
    "win.align-top": f"{PRIMARY_ACCEL}<Shift>Up",
    "win.align-bottom": f"{PRIMARY_ACCEL}<Shift>Down",
    "win.align-h-center": f"{PRIMARY_ACCEL}<Shift>Home",
    "win.align-v-center": f"{PRIMARY_ACCEL}<Shift>End",
    "win.spread-h": f"{PRIMARY_ACCEL}<Shift>h",
    "win.spread-v": f"{PRIMARY_ACCEL}<Shift>v",
    "win.flip-horizontal": "<Shift>h",
    "win.flip-vertical": "<Shift>v",
    # Machine & Help
    "win.machine-settings": f"{PRIMARY_ACCEL}less",
    "win.about": "F1",
}


ActionSetupHandler = Callable[["ActionManager"], None]
ActionStateUpdateHandler = Callable[["ActionManager"], None]


class ActionExtensionRegistry:
    """
    Registry for action extension handlers.

    Allows modules to register their own actions and state update
    handlers, enabling decoupling of functionality like the sketcher.
    """

    def __init__(self):
        self._setup_handlers: List[ActionSetupHandler] = []
        self._state_update_handlers: List[ActionStateUpdateHandler] = []
        self._setup_addon_map: Dict[str, str] = {}
        self._state_update_addon_map: Dict[str, str] = {}

    def register_setup(self, handler: ActionSetupHandler, addon_name: str):
        """Register a handler to be called during action setup."""
        self._setup_handlers.append(handler)
        if addon_name:
            self._setup_addon_map[handler.__name__] = addon_name
        logger.debug(f"Registered action setup handler: {handler.__name__}")

    def unregister_setup(self, handler: ActionSetupHandler) -> bool:
        """Unregister an action setup handler."""
        try:
            self._setup_handlers.remove(handler)
            self._setup_addon_map.pop(handler.__name__, None)
            return True
        except ValueError:
            return False

    def register_state_update(
        self, handler: ActionStateUpdateHandler, addon_name: str
    ):
        """Register a handler to be called during action state updates."""
        self._state_update_handlers.append(handler)
        if addon_name:
            self._state_update_addon_map[handler.__name__] = addon_name
        logger.debug(
            f"Registered action state update handler: {handler.__name__}"
        )

    def unregister_state_update(
        self, handler: ActionStateUpdateHandler
    ) -> bool:
        """Unregister an action state update handler."""
        try:
            self._state_update_handlers.remove(handler)
            self._state_update_addon_map.pop(handler.__name__, None)
            return True
        except ValueError:
            return False

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all handlers registered by a specific addon.

        Args:
            addon_name: The name of the addon to clean up

        Returns:
            The number of handlers unregistered
        """
        count = 0

        setup_to_remove = [
            h
            for h in self._setup_handlers
            if self._setup_addon_map.get(h.__name__) == addon_name
        ]
        for h in setup_to_remove:
            self._setup_handlers.remove(h)
            self._setup_addon_map.pop(h.__name__, None)
        count += len(setup_to_remove)

        state_to_remove = [
            h
            for h in self._state_update_handlers
            if self._state_update_addon_map.get(h.__name__) == addon_name
        ]
        for h in state_to_remove:
            self._state_update_handlers.remove(h)
            self._state_update_addon_map.pop(h.__name__, None)
        count += len(state_to_remove)

        if count > 0:
            logger.debug(
                f"Unregistered {count} action extension handlers "
                f"from addon '{addon_name}'"
            )
        return count

    def invoke_setup_handlers(self, action_manager: "ActionManager"):
        """Invoke all registered setup handlers."""
        for handler in self._setup_handlers:
            try:
                handler(action_manager)
            except Exception as e:
                logger.error(
                    f"Error in action setup handler {handler.__name__}: {e}",
                    exc_info=True,
                )

    def invoke_state_update_handlers(self, action_manager: "ActionManager"):
        """Invoke all registered state update handlers."""
        for handler in self._state_update_handlers:
            try:
                handler(action_manager)
            except Exception as e:
                logger.error(
                    f"Error in action state update handler "
                    f"{handler.__name__}: {e}",
                    exc_info=True,
                )


action_extension_registry = ActionExtensionRegistry()


class ActionManager:
    """Manages the creation and state of all Gio.SimpleActions for the app."""

    def __init__(self, win: "MainWindow"):
        self.win = win
        self.actions: Dict[str, Gio.SimpleAction] = {}
        self._shortcut_controller: Optional[Gtk.ShortcutController] = None
        self._layout_shortcuts: List[Gtk.Shortcut] = []
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
        self._add_action("new", self.win.project_cmd.on_new_project)
        self._add_action("open", self.win.project_cmd.on_open_project)
        self._add_action("save", self.win.project_cmd.on_save_project)
        self._add_action("save-as", self.win.project_cmd.on_save_project_as)
        self._add_action(
            "export_document", self.win.on_export_document_clicked
        )
        # New actions for recent files
        self._add_action(
            "open-recent",
            self.win.project_cmd.on_open_recent,
            GLib.VariantType.new("s"),
        )
        self._add_action("import", self.win.on_menu_import)
        self._add_action("export", self.win.on_export_clicked)
        self._add_action("export-object", self.win.on_export_object_clicked)
        self._add_action("about", self.win.show_about_dialog)
        self._add_action("donate", self.win.on_donate_clicked)
        self._add_action("save_debug_log", self.win.on_save_debug_log)
        self._add_action("settings", self.win.show_settings)
        self._add_action("machine-settings", self.win.show_machine_settings)

        # View Actions
        cv = get_context().config.canvas_view
        self._add_stateful_action(
            "show_3d_view",
            self.win.on_show_3d_view,
            GLib.Variant.new_boolean(False),
        )
        self._add_stateful_action(
            "show_workpieces",
            self.win.on_show_workpieces_state_change,
            GLib.Variant.new_boolean(cv.show_workpieces),
        )
        self._add_stateful_action(
            "toggle_camera_view",
            self.win.on_toggle_camera_view_state_change,
            GLib.Variant.new_boolean(cv.show_camera),
        )
        self._add_stateful_action(
            "toggle_travel_view",
            self.win.on_toggle_travel_view_state_change,
            GLib.Variant.new_boolean(cv.show_travel_lines),
        )
        self._add_stateful_action(
            "show_nogo_zones",
            self.win.on_show_nogo_zones_state_change,
            GLib.Variant.new_boolean(cv.show_nogo_zones),
        )
        self._add_stateful_action(
            "show_models",
            self.win.on_show_models_state_change,
            GLib.Variant.new_boolean(cv.show_models),
        )
        config = get_context().config
        self._add_stateful_action(
            "toggle_bottom_panel",
            self.win.on_toggle_bottom_panel_state_change,
            GLib.Variant.new_boolean(
                config.bottom_panel.get("visible", False)
                if config.bottom_panel
                else False
            ),
        )
        self._add_stateful_action(
            "toggle_right_panel",
            self.win.on_toggle_right_panel_state_change,
            GLib.Variant.new_boolean(config.right_panel_visible),
        )

        # 3D View Control Actions
        self._add_action("view_top", self.win.on_view_top)
        self._add_action("view_front", self.win.on_view_front)
        self._add_action("view_right", self.win.on_view_right)
        self._add_action("view_left", self.win.on_view_left)
        self._add_action("view_back", self.win.on_view_back)
        self._add_action("view_iso", self.win.on_view_iso)
        self._add_stateful_action(
            "view_toggle_perspective",
            self.win.on_view_perspective_state_change,
            GLib.Variant.new_boolean(cv.perspective_mode),
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

        self._add_action("recalculate", self.win.on_recalculate_clicked)
        self._add_action(
            "force-recalculate", self.win.on_force_recalculate_clicked
        )

        # Asset Actions
        self._add_action("add-stock", self.on_add_stock)
        action_registry.register(
            action_name="add-stock",
            action=self.actions["add-stock"],
            addon_name="core",
            label=_("Add Stock"),
            icon_name="stock-symbolic",
            menu=MenuPlacement(menu_id="object", priority=40),
        )
        self._add_action(
            "activate-stock",
            self.on_activate_stock,
            GLib.VariantType.new("s"),
        )
        self._add_action(
            "edit-stock-item",
            self.on_edit_stock_item,
            GLib.VariantType.new("s"),
        )

        # Layer Management Actions
        self._add_action("layer-move-up", self.on_layer_move_up)
        self._add_action("layer-move-down", self.on_layer_move_down)

        # Grouping Actions
        self._add_action("group", self.on_group_action)
        self._add_action("ungroup", self.on_ungroup_action)

        # Split Action
        self._add_action("split", self.on_split_action)

        # Convert to Stock Action
        self._add_action("convert-to-stock", self.on_convert_to_stock)

        # Tabbing Actions
        self._add_action("add-tabs-equidistant", self.on_add_tabs_equidistant)
        self._add_action("add-tabs-cardinal", self.on_add_tabs_cardinal)
        self._add_action("tab-add", self.on_tab_add)
        self._add_action("tab-remove", self.on_tab_remove)
        self._add_stateful_action(
            "show_tabs",
            self.on_show_tabs_state_change,
            GLib.Variant.new_boolean(cv.show_tabs),
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

        self._register_layout_actions()

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

        self._add_action(
            "zero-here",
            self.win.on_zero_here_clicked,
            GLib.VariantType.new("s"),
        )

        action_extension_registry.invoke_setup_handlers(self)

        self.update_action_states()

    def _register_layout_actions(self):
        """Register layout strategy actions with menu and toolbar placement."""
        for strategy_class in layout_registry.list_all():
            name = layout_registry.list_names()[
                list(layout_registry.list_all()).index(strategy_class)
            ]
            if name == "pixel-perfect":
                action = Gio.SimpleAction.new("layout-pixel-perfect", None)
                action.connect("activate", self.on_layout_pixel_perfect)
                action_registry.register(
                    action_name="layout-pixel-perfect",
                    action=action,
                    addon_name="core",
                    label=_("Auto Layout (Simple)"),
                    icon_name="auto-layout-symbolic",
                    shortcut="<Ctrl><Alt>a",
                    menu=MenuPlacement(menu_id="arrange"),
                    toolbar=ToolbarPlacement(group="arrange"),
                )

    def update_action_states(self, *args, **kwargs):
        """Updates the enabled state of actions based on document state."""
        self.actions["add-stock"].set_enabled(True)

        is_unsaved = not self.editor.is_saved
        self.actions["save"].set_enabled(is_unsaved)

        target_workpieces = self._get_workpieces_for_tabbing()
        can_add_tabs = any(wp.boundaries for wp in target_workpieces)
        self.actions["add-tabs-equidistant"].set_enabled(can_add_tabs)
        self.actions["add-tabs-cardinal"].set_enabled(can_add_tabs)

        context = self.win.surface.right_click_context
        can_add_single_tab = context and context.get("type") == "geometry"
        can_remove_single_tab = context and context.get("type") == "tab"
        self.actions["tab-add"].set_enabled(bool(can_add_single_tab))
        self.actions["tab-remove"].set_enabled(bool(can_remove_single_tab))

        selected_wps = self.win.surface.get_selected_workpieces()
        if selected_wps:
            has_workpieces = True
        else:
            current_layer = self.doc.active_layer
            has_workpieces = (
                current_layer
                and len(current_layer.get_descendants(WorkPiece)) > 0
            )
        layout_info = action_registry.get("layout-pixel-perfect")
        if layout_info and layout_info.action:
            layout_info.action.set_enabled(has_workpieces)

        self.actions["split"].set_enabled(bool(selected_wps))
        self.actions["export-object"].set_enabled(len(selected_wps) == 1)

        action_extension_registry.invoke_state_update_handlers(self)

    def on_add_stock(self, action, param):
        """Handler for the 'add-stock' action."""
        self.editor.stock.add_stock()

    def on_activate_stock(self, action, param):
        """Handler for the 'activate-stock' action."""
        asset_uid = param.get_string()
        for item in self.editor.doc.stock_items:
            if item.stock_asset_uid == asset_uid:
                dialog = StockPropertiesDialog(self.win, item, self.editor)
                dialog.present()
                break

    def on_edit_stock_item(self, action, param):
        """Handler for the 'edit-stock-item' action."""
        item_uid = param.get_string()
        item = self.doc.find_descendant_by_uid(item_uid)
        if isinstance(item, StockItem):
            dialog = StockPropertiesDialog(self.win, item, self.editor)
            dialog.present()

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
        self.win.surface.set_global_tab_visibility(is_visible)
        action.set_state(state)
        config = get_context().config
        config.canvas_view.show_tabs = is_visible
        config.changed.send(config)

    def register_shortcuts(self, controller: Gtk.ShortcutController):
        """
        Populates the given ShortcutController with all application shortcuts.
        """
        for action_name, shortcut_str in SHORTCUTS.items():
            shortcut = Gtk.Shortcut.new(
                Gtk.ShortcutTrigger.parse_string(shortcut_str),
                Gtk.NamedAction.new(action_name),
            )
            controller.add_shortcut(shortcut)

        self._shortcut_controller = controller
        self._update_dynamic_shortcuts()

        action_registry.changed.connect(self._on_action_registry_changed)

    def _on_action_registry_changed(self, sender):
        """Handle action registry changes by refreshing shortcuts."""
        self._update_dynamic_shortcuts()

    def _update_dynamic_shortcuts(self):
        """Update shortcuts for actions registered via action_registry."""
        if not self._shortcut_controller:
            return

        for shortcut in self._layout_shortcuts:
            self._shortcut_controller.remove_shortcut(shortcut)
        self._layout_shortcuts.clear()

        for info in action_registry.get_all_with_shortcuts():
            if info.shortcut:
                shortcut = Gtk.Shortcut.new(
                    Gtk.ShortcutTrigger.parse_string(info.shortcut),
                    Gtk.NamedAction.new(f"win.{info.action_name}"),
                )
                self._shortcut_controller.add_shortcut(shortcut)
                self._layout_shortcuts.append(shortcut)

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

    def on_convert_to_stock(self, action, param):
        """Handler for the 'convert-to-stock' action."""
        selected_wps = self.win.surface.get_selected_workpieces()
        if not selected_wps:
            return

        for wp in selected_wps:
            self.editor.stock.convert_to_stock(wp)

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
