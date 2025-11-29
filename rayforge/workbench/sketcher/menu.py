from gi.repository import Gio


class SketchMenu(Gio.Menu):
    """
    The menu model for the Sketch Studio mode.
    """

    def __init__(self):
        super().__init__()

        # File
        file_menu = Gio.Menu()
        file_menu.append(_("Finish Sketch"), "sketch.finish")
        file_menu.append(_("Cancel Sketch"), "sketch.cancel")
        self.append_submenu(_("_File"), file_menu)

        # Edit
        edit_menu = Gio.Menu()
        history_group = Gio.Menu()
        history_group.append(_("Undo"), "sketch.undo")
        history_group.append(_("Redo"), "sketch.redo")
        edit_menu.append_section(None, history_group)

        edit_ops = Gio.Menu()
        edit_ops.append(_("Delete"), "sketch.delete")
        edit_menu.append_section(None, edit_ops)
        self.append_submenu(_("_Edit"), edit_menu)

        # Tools
        tools_menu = Gio.Menu()

        tools_group = Gio.Menu()
        tools_group.append(_("Select"), "sketch.tool_select")
        tools_group.append(_("Line"), "sketch.tool_line")
        tools_group.append(_("Circle"), "sketch.tool_circle")
        tools_group.append(_("Arc"), "sketch.tool_arc")
        tools_menu.append_section(_("Tools"), tools_group)

        constr_group = Gio.Menu()
        constr_group.append(
            _("Toggle Construction"), "sketch.toggle_construction"
        )
        tools_menu.append_section(_("Modify"), constr_group)

        self.append_submenu(_("_Sketch"), tools_menu)

        # View
        view_menu = Gio.Menu()
        view_menu.append(_("Fit View"), "sketch.view_fit")
        self.append_submenu(_("_View"), view_menu)
