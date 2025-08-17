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
        file_menu.append(_("Import..."), "win.import")
        file_menu.append(_("Export G-code..."), "win.export")

        quit_command = Gio.Menu()
        quit_command.append(_("Quit"), "win.quit")
        file_menu.append_section(None, quit_command)
        self.append_submenu(_("_File"), file_menu)

        # Edit Menu
        edit_menu = Gio.Menu()
        edit_menu.append(_("Undo"), "win.undo")
        edit_menu.append(_("Redo"), "win.redo")

        clipboard_commands = Gio.Menu()
        clipboard_commands.append(_("Cut"), "win.cut")
        clipboard_commands.append(_("Copy"), "win.copy")
        clipboard_commands.append(_("Paste"), "win.paste")
        clipboard_commands.append(_("Duplicate"), "win.duplicate")
        clipboard_commands.append(_("Remove"), "win.remove")
        edit_menu.append_section(None, clipboard_commands)

        grouping_commands = Gio.Menu()
        grouping_commands.append(_("Group"), "win.group")
        grouping_commands.append(_("Ungroup"), "win.ungroup")
        edit_menu.append_section(None, grouping_commands)

        other_edit_commands = Gio.Menu()
        other_edit_commands.append(_("Preferences…"), "win.preferences")
        edit_menu.append_section(None, other_edit_commands)
        self.append_submenu(_("_Edit"), edit_menu)

        # View Menu
        view_menu = Gio.Menu()
        view_menu.append(_("Show Workpieces"), "win.show_workpieces")
        view_menu.append_section(None, Gio.Menu.new())  # Separator
        view_menu.append(_("3D View"), "win.show_3d_view")

        view_3d_commands = Gio.Menu()
        view_3d_commands.append(_("Top View"), "win.view_top")
        view_3d_commands.append(_("Front View"), "win.view_front")
        view_3d_commands.append(_("Isometric View"), "win.view_iso")
        view_3d_commands.append(
            _("Toggle Perspective"), "win.view_toggle_perspective"
        )
        view_menu.append_section(None, view_3d_commands)
        self.append_submenu(_("_View"), view_menu)

        # Help Menu
        help_menu = Gio.Menu()
        help_menu.append(_("About"), "win.about")
        self.append_submenu(_("_Help"), help_menu)
