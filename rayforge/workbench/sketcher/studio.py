import logging
from gi.repository import Gtk, Gio, Adw
from blinker import Signal
from ...core.sketcher import Sketch
from ...core.varset import IntVar, FloatVar, SliderFloatVar
from ...shared.ui.varset_editor import VarSetEditorWidget
from ...undo.models.property_cmd import ChangePropertyCommand
from .menu import SketchMenu
from .sketchcanvas import SketchCanvas

logger = logging.getLogger(__name__)


class SketchStudio(Gtk.Box):
    """
    The top-level container for the sketching environment.
    Manages the layout of the canvas and side panels and orchestrates the
    save/cancel lifecycle.
    """

    def __init__(
        self,
        parent_window: Gtk.Window,
        width_mm: float = 1000.0,
        height_mm: float = 1000.0,
        **kwargs,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)
        self.parent_window = parent_window
        self.width_mm = width_mm
        self.height_mm = height_mm

        # Signals
        self.finished = Signal()
        self.cancelled = Signal()

        self._build_ui()

    def set_world_size(self, width_mm: float, height_mm: float):
        """
        Updates the world dimensions of the sketch canvas. This is called
        by the main window when the machine configuration changes.
        """
        self.width_mm = width_mm
        self.height_mm = height_mm
        if self.canvas:
            self.canvas.set_size(width_mm, height_mm)

    def _build_ui(self):
        # 1. Session Bar (Header)
        # We use a Gtk.Box styled as a toolbar to hold the session controls.
        self.session_bar = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6
        )
        self.session_bar.add_css_class("toolbar")
        self.session_bar.set_margin_top(6)
        self.session_bar.set_margin_bottom(6)
        self.session_bar.set_margin_start(12)
        self.session_bar.set_margin_end(12)
        self.append(self.session_bar)

        # Left: Cancel
        self.btn_cancel = Gtk.Button(icon_name="window-close-symbolic")
        self.btn_cancel.set_tooltip_text(_("Cancel Sketch"))
        self.btn_cancel.connect("clicked", self._on_cancel_clicked)
        self.session_bar.append(self.btn_cancel)

        # Center: Title (using spacers to center it roughly)
        spacer_l = Gtk.Box()
        spacer_l.set_hexpand(True)
        self.session_bar.append(spacer_l)

        # We need a vertical box to stack Title and Subtitle
        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        title_box.set_valign(Gtk.Align.CENTER)

        self.lbl_title = Gtk.Label(label=_("Sketch Studio"))
        self.lbl_title.add_css_class("title-3")
        title_box.append(self.lbl_title)

        self.lbl_subtitle = Gtk.Label(
            label=_(
                "Right-click background to draw, objects to edit. "
                "Double-click dimensions to edit values."
            )
        )
        self.lbl_subtitle.add_css_class("caption")
        title_box.append(self.lbl_subtitle)

        self.session_bar.append(title_box)

        spacer_r = Gtk.Box()
        spacer_r.set_hexpand(True)
        self.session_bar.append(spacer_r)

        # Right: Finish
        self.btn_finish = Gtk.Button(label=_("Finish"))
        self.btn_finish.add_css_class("suggested-action")
        self.btn_finish.connect("clicked", self._on_finish_clicked)
        self.session_bar.append(self.btn_finish)

        # 2. Main Content Area (Side Panel + Canvas)
        # Use a Paned widget to allow resizing between the panel and canvas
        main_paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        main_paned.set_vexpand(True)
        main_paned.set_position(450)
        self.append(main_paned)

        # 2a. Side Panel
        side_panel_scroller = Gtk.ScrolledWindow()
        side_panel_scroller.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        # The paned widget handles sizing, no size request is needed here.
        main_paned.set_start_child(side_panel_scroller)

        side_panel_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=12
        )
        side_panel_box.set_margin_top(12)
        side_panel_box.set_margin_bottom(12)
        side_panel_box.set_margin_start(12)
        side_panel_box.set_margin_end(12)
        side_panel_scroller.set_child(side_panel_box)

        # Properties Group (Name)
        properties_group = Adw.PreferencesGroup()
        properties_group.set_title(_("Properties"))
        self.name_row = Adw.EntryRow(title=_("Name"))
        self.name_row.connect("changed", self._on_name_changed)
        properties_group.add(self.name_row)
        side_panel_box.append(properties_group)

        # VarSet Editor Group
        # Limit the types of variables the user can add to the sketch
        self.varset_editor = VarSetEditorWidget(
            vartypes={IntVar, FloatVar, SliderFloatVar}
        )
        self.varset_editor.set_title(_("Input Parameters"))
        side_panel_box.append(self.varset_editor)

        # 2b. Canvas
        self.canvas = SketchCanvas(
            parent_window=self.parent_window,
            single_mode=True,
            width_mm=self.width_mm,
            height_mm=self.height_mm,
        )
        # The paned widget will handle expansion.
        main_paned.set_end_child(self.canvas)

        # Connect history manager to varset editor for undo support
        if self.canvas.sketch_editor:
            self.varset_editor.undo_manager = (
                self.canvas.sketch_editor.history_manager
            )

        # Initialize the VarSetEditor with the default sketch's parameters
        # This ensures variables added before 'set_sketch' are attached to
        # the current sketch
        if self.canvas.sketch_element:
            self.varset_editor.populate(
                self.canvas.sketch_element.sketch.input_parameters
            )

        # 3. Initialize Actions and Menus
        self._init_menu()
        self._init_actions()

    def _init_menu(self):
        """Initializes the menu model."""
        self.menu_model = SketchMenu()

    def _init_actions(self):
        """Initializes the action group and shortcut controller."""
        self.action_group = Gio.SimpleActionGroup()

        # Define actions mapping: (name, callback)
        actions = [
            ("finish", self._on_finish_clicked),
            ("cancel", self._on_cancel_clicked),
            ("undo", self._on_undo),
            ("redo", self._on_redo),
            ("delete", self._on_delete),
            ("view_fit", self._on_view_fit),
            ("toggle_construction", self._on_toggle_construction),
        ]

        for name, cb in actions:
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", cb)
            self.action_group.add_action(action)

        # Tool selection actions
        tool_map = {
            "tool_select": "select",
            "tool_line": "line",
            "tool_circle": "circle",
            "tool_arc": "arc",
        }
        for action_name, tool_id in tool_map.items():
            action = Gio.SimpleAction.new(action_name, None)
            action.connect(
                "activate",
                lambda a, p, t=tool_id: self.canvas.sketch_element.set_tool(t),
            )
            self.action_group.add_action(action)

        # Create shortcut controller
        self.shortcut_controller = Gtk.ShortcutController()
        self.shortcut_controller.set_scope(Gtk.ShortcutScope.MANAGED)

        shortcuts = {
            "sketch.undo": ["<Primary>z"],
            "sketch.redo": ["<Primary>y", "<Primary><Shift>z"],
            "sketch.delete": ["Delete"],
            "sketch.view_fit": ["f"],
            "sketch.tool_select": ["s"],
            "sketch.tool_line": ["l"],
            "sketch.tool_circle": ["c"],
            "sketch.tool_arc": ["a"],
            "sketch.finish": ["<Primary>Return"],
        }

        for action_name, accels in shortcuts.items():
            for accel in accels:
                shortcut = Gtk.Shortcut.new(
                    Gtk.ShortcutTrigger.parse_string(accel),
                    Gtk.NamedAction.new(action_name),
                )
                self.shortcut_controller.add_shortcut(shortcut)

    def set_sketch(self, sketch: Sketch):
        """Loads a sketch model into the studio."""
        logger.debug(
            f"Called with sketch '{sketch.name}' (id: {id(sketch)}) and "
            f"VarSet (id: {id(sketch.input_parameters)})"
        )
        # Load sketch into the canvas element
        self.canvas.set_sketch(sketch)

        # Populate side panel with sketch data
        self.name_row.set_text(sketch.name)
        self.varset_editor.populate(sketch.input_parameters)

        # Ensure the element bounds are updated for the new content
        self.canvas.sketch_element.update_bounds_from_sketch()
        # Reset view to center content
        self.canvas.reset_view()

    # --- Action Handlers ---

    def _on_name_changed(self, entry_row: Adw.EntryRow):
        """Updates the sketch's name with undo support."""
        if not self.canvas or not self.canvas.sketch_element:
            return
        sketch = self.canvas.sketch_element.sketch
        new_name = entry_row.get_text()

        if sketch.name == new_name:
            return

        if self.canvas.sketch_editor:
            cmd = ChangePropertyCommand(
                target=sketch,
                property_name="name",
                new_value=new_name,
                on_change_callback=self._sync_name_ui,
                name=_("Rename Sketch"),
            )
            self.canvas.sketch_editor.history_manager.execute(cmd)
        else:
            sketch.name = new_name

    def _sync_name_ui(self):
        """Updates the UI to match the underlying sketch object (for Undo)."""
        if not self.canvas or not self.canvas.sketch_element:
            return
        sketch = self.canvas.sketch_element.sketch
        if self.name_row.get_text() != sketch.name:
            self.name_row.set_text(sketch.name)

    def _on_undo(self, action, param):
        if self.canvas.sketch_editor:
            self.canvas.sketch_editor.history_manager.undo()

    def _on_redo(self, action, param):
        if self.canvas.sketch_editor:
            self.canvas.sketch_editor.history_manager.redo()

    def _on_delete(self, action, param):
        if self.canvas.sketch_element:
            self.canvas.sketch_element.delete_selection()

    def _on_view_fit(self, action, param):
        self.canvas.reset_view()

    def _on_toggle_construction(self, action, param):
        if self.canvas.sketch_element:
            self.canvas.sketch_element.toggle_construction_on_selection()

    def _on_cancel_clicked(self, btn, *args):
        logger.info("SketchStudio: Cancel clicked")
        self.cancelled.send(self)

    def _on_finish_clicked(self, btn, *args):
        logger.info("SketchStudio: Finish clicked")
        # Retrieve the modified sketch from the canvas element
        if self.canvas.sketch_element:
            sketch = self.canvas.sketch_element.sketch
            self.finished.send(self, sketch=sketch)
