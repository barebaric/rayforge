from gi.repository import Gio


class MainMenu(Gio.Menu):
    """
    The main application menu model, inheriting from Gio.Menu.
    Its constructor builds the entire menu structure.
    """

    def __init__(self):
        super().__init__()

        # File Menu
        file_menu = Gio.Menu()
        file_io_group = Gio.Menu()
        file_io_group.append(_("Import..."), "win.import")
        file_io_group.append(_("Export G-code..."), "win.export")
        file_menu.append_section(None, file_io_group)

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
        settings_group.append(_("Preferences…"), "win.preferences")
        edit_menu.append_section(None, settings_group)
        self.append_submenu(_("_Edit"), edit_menu)

        # View Menu
        view_menu = Gio.Menu()
        visibility_group = Gio.Menu()
        visibility_group.append(_("Show Workpieces"), "win.show_workpieces")
        visibility_group.append(_("Show Tabs"), "win.show_tabs")
        visibility_group.append(
            _("Show Camera Image"), "win.toggle_camera_view"
        )
        visibility_group.append(
            _("Show Travel Moves"), "win.toggle_travel_view"
        )
        view_menu.append_section(None, visibility_group)

        view_3d_group = Gio.Menu()
        view_3d_group.append(_("3D View"), "win.show_3d_view")
        view_menu.append_section(None, view_3d_group)

        view_3d_commands = Gio.Menu()
        view_3d_commands.append(_("Top View"), "win.view_top")
        view_3d_commands.append(_("Front View"), "win.view_front")
        view_3d_commands.append(_("Isometric View"), "win.view_iso")
        view_3d_commands.append(
            _("Toggle Perspective"), "win.view_toggle_perspective"
        )
        view_menu.append_section(None, view_3d_commands)
        self.append_submenu(_("_View"), view_menu)

        # Object Menu
        object_menu = Gio.Menu()
        stock_group = Gio.Menu()
        stock_group.append(_("Add Stock"), "win.add_stock")
        object_menu.append_section(None, stock_group)

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

        layout_group = Gio.Menu()
        layout_group.append(_("Auto Layout"), "win.layout-pixel-perfect")
        arrange_menu.append_section(None, layout_group)
        self.append_submenu(_("Arrange"), arrange_menu)

        # Machine Menu
        machine_menu = Gio.Menu()
        jog_group = Gio.Menu()
        jog_group.append(_("Home"), "win.home")
        jog_group.append(_("Frame"), "win.frame")
        machine_menu.append_section(None, jog_group)

        job_group = Gio.Menu()
        job_group.append(_("Send Job"), "win.send")
        job_group.append(_("Pause / Resume Job"), "win.hold")
        job_group.append(_("Cancel Job"), "win.cancel")
        job_group.append(_("Clear Alarm"), "win.clear_alarm")
        machine_menu.append_section(None, job_group)

        machine_settings_group = Gio.Menu()
        machine_settings_group.append(
            _("Machine Settings…"), "win.machine_settings"
        )
        machine_menu.append_section(None, machine_settings_group)
        self.append_submenu(_("_Machine"), machine_menu)

        # Help Menu
        help_menu = Gio.Menu()
        help_menu.append(_("About"), "win.about")
        self.append_submenu(_("_Help"), help_menu)
