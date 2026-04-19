from gi.repository import Gio, Gtk, GLib
from typing import List
from gettext import gettext as _
from .action_registry import action_registry
from ..machine.models.macro import Macro


class MainMenu(Gio.Menu):
    """
    The main application menu model, inheriting from Gio.Menu.
    Its constructor builds the entire menu structure.
    """

    def __init__(self):
        super().__init__()

        # Store references to menus that can have addon items
        self._addon_sections = {}

        # File Menu
        file_menu = Gio.Menu()
        file_io_group = Gio.Menu()
        file_io_group.append(_("New"), "win.new")
        file_io_group.append(_("Open..."), "win.open")
        file_io_group.append(_("Save"), "win.save")
        file_io_group.append(_("Save As..."), "win.save-as")
        file_menu.append_section(None, file_io_group)

        # New "Open Recent" submenu
        self.recent_files_menu = Gio.Menu()
        self.dynamic_recent_files_section = Gio.Menu()
        self.recent_files_menu.append_section(
            None, self.dynamic_recent_files_section
        )
        file_menu.append_submenu(_("Open Recent"), self.recent_files_menu)

        import_export_group = Gio.Menu()
        import_export_group.append(_("Import..."), "win.import")
        import_export_group.append(_("Export G-code..."), "win.export")
        import_export_group.append(
            _("Export Document..."), "win.export_document"
        )
        file_menu.append_section(None, import_export_group)

        quit_group = Gio.Menu()
        quit_group.append(_("Quit"), "win.quit")
        file_menu.append_section(None, quit_group)
        self.append_submenu(_("_File"), file_menu)

        # Edit Menu
        edit_menu = Gio.Menu()
        history_group = Gio.Menu()
        history_group.append(_("Undo"), "win.undo")
        history_group.append(_("Redo"), "win.redo")
        edit_menu.append_section(None, history_group)

        clipboard_group = Gio.Menu()
        clipboard_group.append(_("Cut"), "win.cut")
        clipboard_group.append(_("Copy"), "win.copy")
        clipboard_group.append(_("Paste"), "win.paste")
        clipboard_group.append(_("Duplicate"), "win.duplicate")
        edit_menu.append_section(None, clipboard_group)

        selection_group = Gio.Menu()
        selection_group.append(_("Select All"), "win.select_all")
        selection_group.append(_("Remove"), "win.remove")
        selection_group.append(_("Clear Document"), "win.clear")
        edit_menu.append_section(None, selection_group)

        settings_group = Gio.Menu()
        settings_group.append(_("Settings"), "win.settings")
        edit_menu.append_section(None, settings_group)
        self.append_submenu(_("_Edit"), edit_menu)

        # View Menu
        view_menu = Gio.Menu()
        visibility_group = Gio.Menu()
        visibility_group.append(
            _("Show Right Panel"), "win.toggle_right_panel"
        )
        visibility_group.append(
            _("Show Bottom Panel"), "win.toggle_bottom_panel"
        )
        view_menu.append_section(None, visibility_group)

        self._view_addon_group = visibility_group

        view_group = Gio.Menu()
        view_group.append(_("3D View"), "win.show_3d_view")
        view_menu.append_section(None, view_group)

        view_3d_commands = Gio.Menu()
        view_3d_commands.append(_("Top View"), "win.view_top")
        view_3d_commands.append(_("Front View"), "win.view_front")
        view_3d_commands.append(_("Right View"), "win.view_right")
        view_3d_commands.append(_("Left View"), "win.view_left")
        view_3d_commands.append(_("Back View"), "win.view_back")
        view_3d_commands.append(_("Isometric View"), "win.view_iso")
        view_3d_commands.append(
            _("Toggle Perspective"), "win.view_toggle_perspective"
        )
        view_menu.append_section(None, view_3d_commands)
        self.append_submenu(_("_View"), view_menu)

        # Object Menu
        object_menu = Gio.Menu()

        other_group = Gio.Menu()
        other_group.append(_("Split"), "win.split")
        other_group.append(_("Export Object..."), "win.export-object")
        object_menu.append_section(None, other_group)

        # Addon section for Object menu
        self._addon_sections["object"] = Gio.Menu()
        object_menu.append_section(None, self._addon_sections["object"])

        tab_submenu = Gio.Menu()
        tab_submenu.append(
            _("Add Equidistant Tabs…"), "win.add-tabs-equidistant"
        )
        tab_submenu.append(_("Add Cardinal Tabs"), "win.add-tabs-cardinal")
        object_menu.append_submenu(_("Add Tabs"), tab_submenu)
        self.append_submenu(_("_Object"), object_menu)

        # Arrange Menu
        arrange_menu = Gio.Menu()
        grouping_group = Gio.Menu()
        grouping_group.append(_("Group"), "win.group")
        grouping_group.append(_("Ungroup"), "win.ungroup")
        arrange_menu.append_section(None, grouping_group)

        layer_group = Gio.Menu()
        layer_group.append(
            _("Move Selection to Layer Above"), "win.layer-move-up"
        )
        layer_group.append(
            _("Move Selection to Layer Below"), "win.layer-move-down"
        )
        arrange_menu.append_section(None, layer_group)

        align_submenu = Gio.Menu()
        align_submenu.append(_("Left"), "win.align-left")
        align_submenu.append(_("Right"), "win.align-right")
        align_submenu.append(_("Top"), "win.align-top")
        align_submenu.append(_("Bottom"), "win.align-bottom")
        align_submenu.append(_("Horizontally Center"), "win.align-h-center")
        align_submenu.append(_("Vertically Center"), "win.align-v-center")
        arrange_menu.append_submenu(_("Align"), align_submenu)

        distribute_submenu = Gio.Menu()
        distribute_submenu.append(_("Spread Horizontally"), "win.spread-h")
        distribute_submenu.append(_("Spread Vertically"), "win.spread-v")
        arrange_menu.append_submenu(_("Distribute"), distribute_submenu)

        flip_submenu = Gio.Menu()
        flip_submenu.append(_("Flip Horizontal"), "win.flip-horizontal")
        flip_submenu.append(_("Flip Vertical"), "win.flip-vertical")
        arrange_menu.append_submenu(_("Flip"), flip_submenu)

        self._layout_group = Gio.Menu()
        arrange_menu.append_section(None, self._layout_group)

        self.append_submenu(_("Arrange"), arrange_menu)

        # Tools Menu
        tools_menu = Gio.Menu()

        # Addon section for Tools menu
        self._addon_sections["tools"] = Gio.Menu()
        tools_menu.append_section(None, self._addon_sections["tools"])

        self.append_submenu(_("_Tools"), tools_menu)

        # Machine Menu
        machine_menu = Gio.Menu()
        jog_group = Gio.Menu()
        jog_group.append(_("Home"), "win.machine-home")
        jog_group.append(_("Frame"), "win.machine-frame")
        machine_menu.append_section(None, jog_group)

        # Macros submenu under jog controls
        macros_menu = Gio.Menu()
        # This section will be populated dynamically
        self.dynamic_macros_section = Gio.Menu()
        macros_menu.append_section(None, self.dynamic_macros_section)
        machine_menu.append_submenu(_("Macros"), macros_menu)

        job_group = Gio.Menu()
        job_group.append(_("Send Job"), "win.machine-send")
        job_group.append(_("Pause / Resume Job"), "win.machine-hold")
        job_group.append(_("Cancel Job"), "win.machine-cancel")
        job_group.append(_("Clear Alarm"), "win.machine-clear-alarm")
        machine_menu.append_section(None, job_group)

        machine_settings_group = Gio.Menu()
        machine_settings_group.append(
            _("Machine Settings"), "win.machine-settings"
        )
        machine_menu.append_section(None, machine_settings_group)

        # Addon section for Machine menu
        self._addon_sections["machine"] = Gio.Menu()
        machine_menu.append_section(None, self._addon_sections["machine"])

        self.append_submenu(_("_Machine"), machine_menu)

        # Help Menu
        help_menu = Gio.Menu()
        help_menu.append(_("About"), "win.about")
        help_menu.append(_("Donate"), "win.donate")
        help_menu.append(_("Save Debug Log"), "win.save_debug_log")
        self.append_submenu(_("_Help"), help_menu)

        # Populate addon menu items
        self._populate_addon_items()
        self._populate_layout_group()

        # Connect to action registry changes
        action_registry.changed.connect(self._on_action_registry_changed)

    def _on_action_registry_changed(self, sender):
        """Handle action registry changes by refreshing addon items."""
        self._populate_addon_items()
        self._populate_layout_group()

    def _populate_layout_group(self):
        """Populate layout strategies in the Arrange menu."""
        self._layout_group.remove_all()
        items = action_registry.get_menu_items("arrange")
        for info in items:
            if info.label:
                self._layout_group.append(
                    info.label, f"win.{info.action_name}"
                )

    def _populate_addon_items(self):
        """Populate addon menu items from the action registry."""
        for menu_id, section in self._addon_sections.items():
            section.remove_all()
            items = action_registry.get_menu_items(menu_id)
            for info in items:
                if info.label:
                    menu_item = Gio.MenuItem.new(
                        info.label, f"win.{info.action_name}"
                    )
                    section.append_item(menu_item)

        self._populate_view_addon_items()

    def _populate_view_addon_items(self):
        """Populate view addon items into the visibility group."""
        n_static = 2
        group = self._view_addon_group
        while group.get_n_items() > n_static:
            group.remove(n_static)
        items = action_registry.get_menu_items("view")
        for info in items:
            if info.label:
                group.append(info.label, f"win.{info.action_name}")

    def update_macros_menu(self, macros: List[Macro]):
        """Clears and rebuilds the dynamic macro execution menu items."""
        self.dynamic_macros_section.remove_all()
        for macro in macros:
            action_name = f"win.execute-macro('{macro.uid}')"
            self.dynamic_macros_section.append(macro.name, action_name)

    def update_recent_files_menu(self, recent_infos: List[Gtk.RecentInfo]):
        """Clears and rebuilds the dynamic recent files menu."""
        self.dynamic_recent_files_section.remove_all()

        if not recent_infos:
            # Action is None, so it will appear insensitive
            self.dynamic_recent_files_section.append(
                _("(No Recent Items)"), None
            )
            return

        for info in recent_infos:
            # Use Glib.markup_escape to prevent issues with special chars
            # in filenames.
            display_name = GLib.markup_escape_text(info.get_display_name())
            uri = info.get_uri()
            action_name = f"win.open-recent('{uri}')"
            self.dynamic_recent_files_section.append(display_name, action_name)
