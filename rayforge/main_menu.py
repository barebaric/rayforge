from gi.repository import Gio  # type: ignore


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

        other_edit_commands = Gio.Menu()
        other_edit_commands.append(_("Preferences…"), "win.preferences")
        edit_menu.append_section(None, other_edit_commands)
        self.append_submenu(_("_Edit"), edit_menu)

        # Help Menu
        help_menu = Gio.Menu()
        help_menu.append(_("About"), "win.about")
        self.append_submenu(_("_Help"), help_menu)
