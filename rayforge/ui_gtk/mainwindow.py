import asyncio
import logging
from concurrent.futures import Future
from pathlib import Path
from typing import Callable, Coroutine, List, Optional, cast
from gi.repository import Adw, Gdk, Gio, GLib, Gtk

from .doceditor import file_dialogs
from .. import __version__
from .actions import ActionManager
from ..context import get_context
from ..core.group import Group
from ..core.item import DocItem
from ..core.sketcher import Sketch
from ..core.step import Step
from ..core.stock import StockItem
from ..core.undo import Command, HistoryManager, ListItemCommand
from ..core.workpiece import WorkPiece
from ..doceditor.editor import DocEditor
from .doceditor import import_handler
from .doceditor.asset_list_view import AssetListView
from .doceditor.item_properties import DocItemPropertiesWidget
from .doceditor.layer_list import LayerListView
from .doceditor.stock_properties_dialog import StockPropertiesDialog
from .doceditor.sketch_properties import SketchPropertiesWidget
from .doceditor.workflow_view import WorkflowView
from ..image.sketch.exporter import SketchExporter
from ..machine.cmd import MachineCmd
from ..machine.driver.driver import DeviceState, DeviceStatus
from ..machine.driver.dummy import NoDeviceDriver
from ..machine.models.machine import Machine
from ..machine.transport import TransportStatus
from .machine.jog_dialog import JogDialog
from .machine.log_dialog import MachineLogDialog
from .machine.settings_dialog import MachineSettingsDialog
from .main_menu import MainMenu
from ..pipeline.artifact import JobArtifact, JobArtifactHandle
from ..pipeline.encoder.gcode import MachineCodeOpMap
from ..pipeline.steps import STEP_FACTORIES, create_contour_step
from ..shared.gcodeedit.viewer import GcodeViewer
from ..shared.tasker import task_mgr
from .about import AboutDialog
from .settings.settings_dialog import SettingsWindow
from .task_bar import TaskBar
from .toolbar import MainToolbar
from .canvas import CanvasElement
from .canvas2d.drag_drop_cmd import DragDropCmd
from .canvas2d.elements.stock import StockElement
from .canvas2d.simulator_cmd import SimulatorCmd
from .canvas2d.surface import WorkSurface
from .canvas3d import Canvas3D, initialized as canvas3d_initialized
from .sketcher.cmd import UpdateSketchCommand
from .sketcher.studio import SketchStudio
from .view_mode_cmd import ViewModeCmd


logger = logging.getLogger(__name__)


css = """
.mainpaned > separator {
    border: none;
    box-shadow: none;
}

.statusbar {
    border-radius: 5px;
    padding-top: 6px;
}

.statusbar:hover {
    background-color: alpha(@theme_fg_color, 0.1);
}

.in-header-menubar {
    margin-left: 6px;
    box-shadow: none;
}

.in-header-menubar item {
    padding: 6px 12px 6px 12px;
}

.menu separator {
    border-top: 1px solid @borders;
    margin-top: 5px;
    margin-bottom: 5px;
}

.warning-label {
    color: @warning_color;
    font-weight: bold;
}
"""


def _get_monitor_geometry() -> Optional[Gdk.Rectangle]:
    """
    Returns a rectangle for the current monitor dimensions. If not found,
    may return None.
    """
    display = Gdk.Display.get_default()
    if not display:
        return None

    monitors = display.get_monitors()
    if not monitors:
        return None
    monitor = cast(Gdk.Monitor, monitors[0])

    # Try to get the monitor under the cursor (heuristic for active
    # monitor). Note: Wayland has no concept of "primary monitor"
    # anymore, so Gdk.get_primary_monitor() is obsolete.
    # Fallback to the first monitor if no monitor is found under the cursor
    seat = display.get_default_seat()
    if not seat:
        return monitor.get_geometry()

    pointer = seat.get_pointer()
    if not pointer:
        return monitor.get_geometry()

    surface, x, y = pointer.get_surface_at_position()
    if not surface:
        return monitor.get_geometry()

    monitor_under_mouse = display.get_monitor_at_surface(surface)
    if not monitor_under_mouse:
        return monitor.get_geometry()

    return monitor_under_mouse.get_geometry()


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_title(_("Rayforge"))
        self._current_machine: Optional[Machine] = None  # For signal handling
        self._last_gcode_previewer_width = 350
        self._live_3d_view_connected = False

        # The ToastOverlay will wrap the main content box
        self.toast_overlay = Adw.ToastOverlay()
        self.set_content(self.toast_overlay)
        # Track active toasts so they can be cleared programmatically
        self._active_toasts: List[Adw.Toast] = []

        # The main content box is now the child of the ToastOverlay
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.toast_overlay.set_child(vbox)

        # Create the central document editor. This now owns the Doc and
        # Pipeline.
        context = get_context()
        self.doc_editor = DocEditor(task_mgr, context)
        self.machine_cmd = MachineCmd(self.doc_editor)
        self.machine_cmd.job_started.connect(self._on_job_started)

        # Instantiate UI-specific command handlers
        self.view_cmd = ViewModeCmd(self.doc_editor, self)
        self.simulator_cmd = SimulatorCmd(self)

        # Add a key controller to handle ESC key for exiting simulation mode
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

        geometry = _get_monitor_geometry()
        if geometry:
            self.set_default_size(
                int(geometry.width * 0.8), int(geometry.height * 0.8)
            )
        else:
            self.set_default_size(1100, 800)

        # HeaderBar with left-aligned menu and centered title
        # MOVED: Now direct child of vbox to persist across views
        header_bar = Adw.HeaderBar()
        vbox.append(header_bar)

        # Create the menu model and the popover menubar
        self.menu_model = MainMenu()
        self.menubar = Gtk.PopoverMenuBar.new_from_model(self.menu_model)
        self.menubar.add_css_class("in-header-menubar")
        header_bar.pack_start(self.menubar)

        # Create and set the centered title widget
        window_title = Adw.WindowTitle(
            title=self.get_title() or "", subtitle=__version__ or ""
        )
        header_bar.set_title_widget(window_title)

        # Create a stack for switching between main view and sketch studio
        self.main_stack = Gtk.Stack()
        self.main_stack.set_vexpand(True)
        self.main_stack.set_transition_type(
            Gtk.StackTransitionType.SLIDE_UP_DOWN
        )
        vbox.append(self.main_stack)

        # Create a container for the main UI
        main_ui_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.main_stack.add_named(main_ui_box, "main")

        # Create and add the main toolbar.
        self.toolbar = MainToolbar()
        self._connect_toolbar_signals()
        main_ui_box.append(self.toolbar)

        # Create the Paned splitting the window into left and right sections.
        self.paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self.paned.set_vexpand(True)
        main_ui_box.append(self.paned)

        # Apply styles
        self.paned.add_css_class("mainpaned")
        display = Gdk.Display.get_default()
        if display:
            provider = Gtk.CssProvider()
            provider.load_from_string(css)
            Gtk.StyleContext.add_provider_for_display(
                display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )

        # Determine initial machine dimensions for all canvases.
        config = get_context().config
        if config.machine:
            width_mm, height_mm = config.machine.dimensions
            y_down = config.machine.y_axis_down
            x_right = config.machine.x_axis_right
            reverse_x = config.machine.reverse_x_axis
            reverse_y = config.machine.reverse_y_axis
        else:
            # Default to a square aspect ratio if no machine is configured
            width_mm, height_mm = 100.0, 100.0
            y_down, x_right, reverse_x, reverse_y = (
                False,
                False,
                False,
                False,
            )

        # Create the Sketch Studio, passing the machine dimensions.
        self.sketch_studio = SketchStudio(
            self, width_mm=width_mm, height_mm=height_mm
        )
        self.main_stack.add_named(self.sketch_studio, "sketch")
        self.active_sketch_workpiece: Optional[WorkPiece] = None
        self._is_editing_new_sketch = False
        self.sketch_studio.finished.connect(self._on_sketch_finished)
        self.sketch_studio.cancelled.connect(self._on_sketch_cancelled)

        self.surface = WorkSurface(
            editor=self.doc_editor,
            parent_window=self,
            machine=config.machine,
            cam_visible=True,  # Will be set by action state
        )
        self.surface.set_hexpand(True)

        # Initialize drag-and-drop command for the surface
        self.drag_drop_cmd = DragDropCmd(self, self.surface)
        self.surface.drag_drop_cmd = self.drag_drop_cmd
        self.drag_drop_cmd.setup_drop_targets()

        # Setup keyboard actions using the new ActionManager.
        self.action_manager = ActionManager(self)
        self.action_manager.register_actions()
        shortcut_controller = Gtk.ShortcutController()
        self.action_manager.register_shortcuts(shortcut_controller)
        self.add_controller(shortcut_controller)

        # Set the initial state of the surface based on the action's default
        show_tabs_action = self.action_manager.get_action("show_tabs")
        state = show_tabs_action.get_state()
        initial_state = state.get_boolean() if state else True
        self.surface.set_global_tab_visibility(initial_state)

        # Connect document signals
        doc = self.doc_editor.doc
        self._initialize_document()
        doc.updated.connect(self.on_doc_changed)
        doc.descendant_added.connect(self.on_doc_changed)
        doc.descendant_removed.connect(self.on_doc_changed)
        doc.active_layer_changed.connect(self._on_active_layer_changed)
        doc.history_manager.changed.connect(self.on_history_changed)

        # Connect editor signals
        self.doc_editor.notification_requested.connect(
            self._on_editor_notification
        )
        self.doc_editor.document_settled.connect(self._on_document_settled)

        # Connect to Pipeline signals
        self.doc_editor.pipeline.job_time_updated.connect(
            self._on_job_time_updated
        )

        # Create the view stack for 2D and 3D views
        self.view_stack = Gtk.Stack()
        self.view_stack.set_transition_type(
            Gtk.StackTransitionType.SLIDE_LEFT_RIGHT
        )
        self.view_stack.set_margin_start(12)
        self.view_stack.set_hexpand(True)
        self.view_stack.connect(
            "notify::visible-child-name", self._on_view_stack_changed
        )

        # Create the G-code previewer
        self.gcode_previewer = GcodeViewer()
        self.gcode_previewer.set_size_request(
            self._last_gcode_previewer_width, -1
        )
        self.gcode_previewer.line_activated.connect(
            self._on_gcode_line_activated
        )

        # Create a new paned for the left side of the window
        self.left_content_pane = Gtk.Paned(
            orientation=Gtk.Orientation.HORIZONTAL
        )
        # Put the previewer directly into the paned, NO REVEALER
        self.left_content_pane.set_start_child(self.gcode_previewer)
        self.left_content_pane.set_end_child(self.view_stack)
        self.left_content_pane.set_resize_end_child(True)
        self.left_content_pane.set_shrink_end_child(False)

        # Connect to the position signal to remember the user's chosen width
        self.left_content_pane.connect(
            "notify::position", self._on_left_pane_position_changed
        )
        # Set the initial position to 0 to start "hidden"
        self.left_content_pane.set_position(0)

        # The new left-side paned is the start child of the main paned
        self.paned.set_start_child(self.left_content_pane)

        # Wrap surface in an overlay to allow preview controls
        self.surface_overlay = Gtk.Overlay()
        self.surface_overlay.set_child(self.surface)
        self.view_stack.add_named(self.surface_overlay, "2d")

        # Add a click handler to unfocus when clicking the "dead space" of the
        # canvas area. This is the correct place for this handler, as it won't
        # interfere with clicks on the sidebar.
        canvas_click_gesture = Gtk.GestureClick.new()
        canvas_click_gesture.connect(
            "pressed", self._on_canvas_area_click_pressed
        )
        # self.surface_overlay.add_controller(canvas_click_gesture)

        if canvas3d_initialized:
            self.canvas3d = Canvas3D(
                context,
                self.doc_editor.doc,
                self.doc_editor.pipeline,
                width_mm=width_mm,
                depth_mm=height_mm,
                y_down=y_down,
                x_right=x_right,
                x_negative=reverse_x,
                y_negative=reverse_y,
            )

            # Create a stack to switch between 2D and 3D views
            self.view_stack.add_named(self.canvas3d, "3d")

        # Undo/Redo buttons are now connected to the doc via actions.
        self.toolbar.undo_button.set_history_manager(
            self.doc_editor.history_manager
        )
        self.toolbar.redo_button.set_history_manager(
            self.doc_editor.history_manager
        )

        # Create a vertical paned for the right pane content
        right_pane_scrolled_window = Gtk.ScrolledWindow()
        right_pane_scrolled_window.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        right_pane_scrolled_window.set_vexpand(True)
        right_pane_scrolled_window.set_margin_start(10)
        right_pane_scrolled_window.set_margin_top(6)
        right_pane_scrolled_window.set_margin_bottom(12)
        self.paned.set_end_child(right_pane_scrolled_window)
        self.paned.set_resize_end_child(False)
        self.paned.set_shrink_end_child(False)

        # Create a vertical box to organize the content within the
        # ScrolledWindow.
        right_pane_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        right_pane_box.set_size_request(400, -1)
        right_pane_scrolled_window.set_child(right_pane_box)

        # Add the unified Asset list view
        self.asset_list_view = AssetListView(self.doc_editor)
        self.asset_list_view.set_margin_end(12)
        right_pane_box.append(self.asset_list_view)
        self.asset_list_view.add_sketch_clicked.connect(self.on_new_sketch)
        self.asset_list_view.add_stock_clicked.connect(self.on_add_child)
        self.asset_list_view.sketch_activated.connect(
            self._on_sketch_definition_activated
        )

        # Add the Layer list view
        self.layer_list_view = LayerListView(self.doc_editor)
        self.layer_list_view.set_margin_top(20)
        self.layer_list_view.set_margin_end(12)
        right_pane_box.append(self.layer_list_view)

        # The WorkflowView will be updated when a layer is activated.
        initial_workflow = self.doc_editor.doc.active_layer.workflow
        assert initial_workflow, "Initial active layer must have a workflow"
        self.workflowview = WorkflowView(
            self.doc_editor,
            initial_workflow,
            step_factories=STEP_FACTORIES,
        )
        self.workflowview.set_margin_top(20)
        self.workflowview.set_margin_end(12)
        right_pane_box.append(self.workflowview)

        # Add the WorkpiecePropertiesWidget
        self.item_props_widget = DocItemPropertiesWidget(
            editor=self.doc_editor
        )
        item_props_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.item_props_widget.set_margin_top(20)
        self.item_props_widget.set_margin_end(12)
        item_props_container.append(self.item_props_widget)

        self.item_revealer = Gtk.Revealer()
        self.item_revealer.set_child(item_props_container)
        self.item_revealer.set_reveal_child(False)
        self.item_revealer.set_transition_type(
            Gtk.RevealerTransitionType.SLIDE_UP
        )
        right_pane_box.append(self.item_revealer)

        # Add the SketchPropertiesWidget
        self.sketch_props_widget = SketchPropertiesWidget(
            editor=self.doc_editor
        )
        self.sketch_props_widget.set_margin_top(20)
        self.sketch_props_widget.set_margin_end(12)
        right_pane_box.append(self.sketch_props_widget)

        # Connect signals for item selection and actions
        self.surface.selection_changed.connect(self._on_selection_changed)
        self.surface.elements_deleted.connect(self.on_elements_deleted)
        self.surface.cut_requested.connect(self.on_cut_requested)
        self.surface.copy_requested.connect(self.on_copy_requested)
        self.surface.paste_requested.connect(self.on_paste_requested)
        self.surface.duplicate_requested.connect(self.on_duplicate_requested)
        self.surface.transform_initiated.connect(
            self._on_surface_transform_initiated
        )
        self.surface.transform_end.connect(self._on_surface_transform_end)

        # Connect new signal from WorkSurface for edit sketch requests
        self.surface.edit_sketch_requested.connect(
            self._on_edit_sketch_requested
        )
        self.surface.edit_stock_item_requested.connect(
            self._on_edit_stock_item_requested
        )

        # Create and add the status monitor widget.
        self.status_monitor = TaskBar(task_mgr)
        self.status_monitor.log_requested.connect(self.on_status_bar_clicked)
        main_ui_box.append(self.status_monitor)

        # Set up config signals.
        config.changed.connect(self.on_config_changed)
        task_mgr.tasks_updated.connect(self.on_running_tasks_changed)
        self.needs_homing = (
            config.machine.home_on_start if config.machine else False
        )

        # Set initial state
        self.on_config_changed(None)

    def on_add_child(self, sender):
        """Handler for adding a new stock item, called from AssetListView."""
        self.doc_editor.stock.add_stock()

    def enter_sketch_mode(
        self, workpiece: WorkPiece, is_new_sketch: bool = False
    ):
        """Switches the view to the SketchStudio to edit a workpiece."""
        sketch = None
        if workpiece.sketch_uid:
            sketch = cast(
                Optional["Sketch"],
                self.doc_editor.doc.get_asset_by_uid(workpiece.sketch_uid),
            )

        if not sketch:
            logger.warning("Attempted to edit a non-sketch workpiece.")
            return

        try:
            self.active_sketch_workpiece = workpiece
            self._is_editing_new_sketch = is_new_sketch
            self.sketch_studio.set_sketch(sketch)
            self.main_stack.set_visible_child_name("sketch")

            # Swap menu and actions
            self.menubar.set_menu_model(self.sketch_studio.menu_model)
            self.insert_action_group("sketch", self.sketch_studio.action_group)
            self.add_controller(self.sketch_studio.shortcut_controller)
        except Exception as e:
            logger.error(
                f"Failed to load sketch for editing: {e}", exc_info=True
            )

    def exit_sketch_mode(self):
        """Returns to the main 2D/3D view from the SketchStudio."""
        # Restore menu and actions
        self.menubar.set_menu_model(self.menu_model)
        self.insert_action_group("sketch", None)
        self.remove_controller(self.sketch_studio.shortcut_controller)

        self.main_stack.set_visible_child_name("main")
        self.active_sketch_workpiece = None
        self._is_editing_new_sketch = False

    def enter_sketch_definition_mode(self, sketch: Sketch):
        """Switches to SketchStudio to edit a sketch definition directly."""
        try:
            self.active_sketch_workpiece = None
            self._is_editing_new_sketch = False
            self.sketch_studio.set_sketch(sketch)
            self.main_stack.set_visible_child_name("sketch")

            # Swap menu and actions
            self.menubar.set_menu_model(self.sketch_studio.menu_model)
            self.insert_action_group("sketch", self.sketch_studio.action_group)
            self.add_controller(self.sketch_studio.shortcut_controller)
        except Exception as e:
            logger.error(
                f"Failed to load sketch definition for editing: {e}",
                exc_info=True,
            )

    def _on_sketch_definition_activated(self, sender, *, sketch: Sketch):
        """Handles activation of a sketch definition from the sketch list."""
        self.enter_sketch_definition_mode(sketch)

    def _on_sketch_finished(self, sender, *, sketch: Sketch):
        """Handles the 'finished' signal from the SketchStudio."""
        cmd = UpdateSketchCommand(
            doc=self.doc_editor.doc,
            sketch_uid=sketch.uid,
            new_sketch_dict=sketch.to_dict(),
        )
        self.doc_editor.history_manager.execute(cmd)
        self.exit_sketch_mode()

    def _on_sketch_cancelled(self, sender):
        """Handles the 'cancelled' signal from the SketchStudio."""
        was_new = self._is_editing_new_sketch
        self.exit_sketch_mode()

        if was_new:
            # If we just created this sketch, remove it from the doc by
            # undoing the creation command.
            self.doc_editor.history_manager.undo()

    def on_new_sketch(self, action=None, param=None):
        """Action handler for creating a new sketch definition."""
        # 1. Create a new, empty sketch object
        new_sketch = Sketch(name=_("New Sketch"))

        # 2. Create and execute an undoable command to add it to the document
        command = ListItemCommand(
            owner_obj=self.doc_editor.doc,
            item=new_sketch,
            undo_command="remove_asset",
            redo_command="add_asset",
            name=_("Create Sketch Definition"),
        )
        self.doc_editor.history_manager.execute(command)

        # 3. Immediately enter edit mode for the new definition
        self.enter_sketch_definition_mode(new_sketch)
        self._is_editing_new_sketch = True

    def on_edit_sketch(self, action, param):
        """Action handler for editing the selected sketch."""
        selected_items = self.surface.get_selected_workpieces()
        if len(selected_items) == 1 and isinstance(
            selected_items[0], WorkPiece
        ):
            wp = selected_items[0]
            if wp.sketch_uid:
                self.enter_sketch_mode(wp)
            else:
                self._on_editor_notification(
                    self, _("Selected item is not an editable sketch.")
                )
        else:
            self._on_editor_notification(
                self, _("Please select a single sketch to edit.")
            )

    def _on_edit_sketch_requested(self, sender, *, workpiece: WorkPiece):
        """Signal handler for edit sketch requests from the surface."""
        logger.debug(f"Sketch edit requested for workpiece {workpiece.name}")
        self.enter_sketch_mode(workpiece)

    def _on_edit_stock_item_requested(self, sender, *, stock_item: StockItem):
        """Signal handler for edit stock item requests from the surface."""
        logger.debug(
            f"Stock properties requested for stock item {stock_item.name}"
        )
        dialog = StockPropertiesDialog(self, stock_item, self.doc_editor)
        dialog.present()

    def on_export_sketch(self, action, param):
        """Action handler for exporting the selected sketch."""
        selected_items = self.surface.get_selected_workpieces()
        if len(selected_items) == 1:
            # The ActionManager already validates that this is a sketch-based
            # workpiece
            file_dialogs.show_export_sketch_dialog(
                self, self._on_export_sketch_save_response, selected_items[0]
            )
        else:
            self._on_editor_notification(
                self, _("Please select a single sketch to export.")
            )

    def _on_export_sketch_save_response(self, dialog, result, user_data):
        """Callback for the export sketch dialog."""
        try:
            file = dialog.save_finish(result)
            if not file:
                return
            file_path = Path(file.get_path())

            # Re-verify selection to be safe
            selected = self.surface.get_selected_workpieces()
            if len(selected) != 1:
                return

            wp = selected[0]
            exporter = SketchExporter(wp)
            data = exporter.export()

            try:
                file_path.write_bytes(data)
                self._on_editor_notification(
                    self, _("Sketch exported successfully.")
                )
            except Exception as e:
                logger.error(
                    f"Failed to write sketch file: {e}", exc_info=True
                )
                self._on_editor_notification(
                    self, _("Failed to save sketch file.")
                )

        except GLib.Error as e:
            logger.error(f"Error saving file: {e.message}")
            return
        except ValueError as e:
            logger.error(f"Error exporting sketch: {e}")
            self._on_editor_notification(
                self, _("Error exporting sketch: {error}").format(error=str(e))
            )

    def _update_macros_menu(self, *args):
        """Rebuilds the dynamic 'Macros' menu."""
        config = get_context().config
        if not config.machine:
            self.menu_model.update_macros_menu([])
            return

        macros = sorted(
            config.machine.macros.values(), key=lambda m: m.name.lower()
        )
        enabled_macros = [m for m in macros if m.enabled]
        self.menu_model.update_macros_menu(enabled_macros)

    def on_execute_macro(self, action: Gio.SimpleAction, param: GLib.Variant):
        """Handler for the 'execute-macro' action."""
        config = get_context().config
        if not config.machine:
            return
        macro_uid = param.get_string()
        logger.info(f"Executing macro: {macro_uid}")
        self.machine_cmd.execute_macro_by_uid(config.machine, macro_uid)

    def _on_job_started(self, sender):
        """Handles the start of a machine job."""
        logger.debug("Job started")

        # Determine which view to show based on the machine's capability
        is_granular = False
        config = get_context().config
        if config.machine:
            is_granular = config.machine.reports_granular_progress
        self.status_monitor.start_live_view(is_granular)

    def _on_job_progress_updated(self, metrics: dict):
        """Callback for when job progress is updated."""
        logger.debug(f"Job progress updated: {metrics}")
        self.status_monitor.update_live_progress(metrics)

    def _on_job_finished(self, sender):
        """Handles the completion of a machine job."""
        logger.debug("Job finished")
        self.status_monitor.stop_live_view()

    def _on_job_future_done(self, future: Future):
        """Callback for when the job submission task completes or fails."""
        try:
            # Check for exceptions during job assembly or submission.
            future.result()
        except Exception as e:
            logger.error(f"Job submission failed: {e}", exc_info=True)
            # If the submission failed, the driver's 'job_finished' signal
            # will never fire, so we must stop the live view here to prevent
            # the UI from getting stuck.
            self.status_monitor.stop_live_view()

    def _on_gcode_line_activated(self, sender, *, line_number: int):
        """
        Handles the user activating a line in the G-code previewer.
        Syncs the highlight and the simulation slider.
        """
        # 1. Update the visual highlight to match the cursor, no scroll.
        self.gcode_previewer.highlight_line(line_number, use_align=False)

        # 2. If simulation is active, update its position.
        if self.simulator_cmd.preview_controls:
            self.simulator_cmd.sync_from_gcode(line_number)

    def _on_left_pane_position_changed(self, paned, param):
        """
        Stores the user-defined width of the G-code previewer pane so it can
        be restored later.
        """
        position = paned.get_position()
        # Only store the position if the pane is open. This prevents
        # storing '0' when it gets hidden automatically.
        if position > 1:
            self._last_gcode_previewer_width = position

    def _on_surface_transform_initiated(self, sender):
        """
        Called when the user starts a transform on the canvas. If simulation
        is active, this will turn it off.
        """
        sim_action = self.action_manager.get_action("simulate_mode")
        state = sim_action.get_state() if sim_action else None
        if state and state.get_boolean():
            sim_action.change_state(GLib.Variant.new_boolean(False))

    def _on_view_stack_changed(self, stack: Gtk.Stack, param):
        """Handles logic when switching between 2D and 3D views."""
        child_name = stack.get_visible_child_name()
        if child_name == "3d":
            self._connect_live_3d_view_signals()
        else:
            self._disconnect_live_3d_view_signals()
        self._update_actions_and_ui()

    def _connect_live_3d_view_signals(self):
        """Connects to Pipeline signals to update the 3D view live."""
        if self._live_3d_view_connected:
            return
        logger.debug("Connecting live 3D view signals.")
        gen = self.doc_editor.pipeline
        gen.workpiece_artifact_ready.connect(self._on_live_3d_view_update)
        self._live_3d_view_connected = True
        # Trigger a full update to draw the current state immediately
        self._update_3d_view_content()

    def _disconnect_live_3d_view_signals(self):
        """Disconnects from Pipeline signals."""
        if not self._live_3d_view_connected:
            return
        logger.debug("Disconnecting live 3D view signals.")
        gen = self.doc_editor.pipeline
        gen.workpiece_artifact_ready.disconnect(self._on_live_3d_view_update)
        self._live_3d_view_connected = False

    def _on_live_3d_view_update(
        self,
        sender: Optional[Step],
        *,
        workpiece: Optional[WorkPiece],
        generation_id: int,
    ):
        """
        When an artifact's generation is finished, trigger a full scene update
        for the 3D view. The arguments are optional to allow manual calls.
        """
        if self.view_stack.get_visible_child_name() == "3d":
            self._update_3d_view_content()

    def _update_3d_view_content(self):
        """
        Updates the 3D canvas by delegating to its internal update method.
        This is now a fast, non-blocking operation.
        """
        if not self.canvas3d:
            return
        self.canvas3d.update_scene_from_doc()

    def _update_gcode_preview(
        self, gcode_string: Optional[str], op_map: Optional[MachineCodeOpMap]
    ):
        """Updates the G-code preview panel from a pre-generated string."""
        if gcode_string is None:
            self.gcode_previewer.clear()
            return

        self.gcode_previewer.set_gcode(gcode_string)
        if op_map:
            self.gcode_previewer.set_op_map(op_map)

    def on_show_3d_view(
        self, action: Gio.SimpleAction, value: Optional[GLib.Variant]
    ):
        """Delegates the view switching logic to the command module."""
        self.view_cmd.toggle_3d_view(action, value)

    def on_show_workpieces_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        is_visible = value.get_boolean()
        self.surface.set_workpieces_visible(is_visible)
        action.set_state(value)

    def on_toggle_camera_view_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        is_visible = value.get_boolean()
        self.surface.set_camera_image_visibility(is_visible)
        button = self.toolbar.camera_visibility_button
        if is_visible:
            button.set_child(self.toolbar.camera_visibility_on_icon)
        else:
            button.set_child(self.toolbar.camera_visibility_off_icon)
        action.set_state(value)

    def on_toggle_travel_view_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        is_visible = value.get_boolean()
        self.surface.set_show_travel_moves(is_visible)
        if canvas3d_initialized and hasattr(self, "canvas3d"):
            self.canvas3d.set_show_travel_moves(is_visible)
        action.set_state(value)

    def on_toggle_gcode_preview_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        """Handles the state change for the G-code preview visibility."""
        is_visible = value.get_boolean()
        action.set_state(value)

        if is_visible:
            self.left_content_pane.set_position(
                self._last_gcode_previewer_width
            )
            # The content will be loaded by the 'document-settled' handler.
            # We just need to trigger it in case the doc is already settled.
            self.refresh_previews()
        else:
            self.left_content_pane.set_position(0)

    def on_view_top(self, action, param):
        """Action handler to set the 3D view to top-down."""
        self.view_cmd.set_view_top(self.canvas3d)

    def on_view_front(self, action, param):
        """Action handler to set the 3D view to front."""
        self.view_cmd.set_view_front(self.canvas3d)

    def on_view_iso(self, action, param):
        """Action handler to set the 3D view to isometric."""
        self.view_cmd.set_view_iso(self.canvas3d)

    def on_view_perspective_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        """Handles state changes for the perspective view action."""
        self.view_cmd.toggle_perspective(self.canvas3d, action, value)

    def _initialize_document(self):
        """
        Adds required initial state to a new document, such as a default
        step to the first workpiece layer.
        """
        doc = self.doc_editor.doc
        if not doc.layers:
            return

        first_workpiece_layer = doc.layers[0]
        if (
            first_workpiece_layer
            and first_workpiece_layer.workflow
            and not first_workpiece_layer.workflow.has_steps()
        ):
            workflow = first_workpiece_layer.workflow
            # The first factory in the list is the default step type
            default_step = create_contour_step(get_context())

            # Apply best recipe using the new helper method
            self.doc_editor.step.apply_best_recipe_to_step(default_step)

            workflow.add_step(default_step)
            logger.info(
                f"Added default '{default_step.typelabel}' step to "
                "initial document."
            )

    def _connect_toolbar_signals(self):
        """Connects signals from the MainToolbar to their handlers.
        Most buttons are connected via Gio.Actions. Only view-state toggles
        and special widgets are connected here.
        """
        self.toolbar.machine_warning_clicked.connect(
            self.on_machine_warning_clicked
        )
        self.toolbar.machine_selector.machine_selected.connect(
            self.on_machine_selected_by_selector
        )

    def _on_canvas_area_click_pressed(self, gesture, n_press, x, y):
        """
        Handler for clicks on the canvas overlay area (the 'dead space').
        It unfocuses any other widget and gives focus to the surface for
        keyboard shortcuts.
        """
        logger.debug("Clicked on canvas area dead space, focusing surface.")
        self.surface.grab_focus()

    def on_machine_selected_by_selector(self, sender, *, machine: Machine):
        """
        Handles the 'machine_selected' signal from the MachineSelector widget,
        delegating the logic to the MachineManager.
        """
        context = get_context()
        context.machine_mgr.set_active_machine(machine)

    def _on_machine_status_changed(self, machine: Machine, state: DeviceState):
        """Called when the active machine's state changes."""
        config = get_context().config
        if self.needs_homing and config.machine and config.machine.driver:
            if state.status == DeviceStatus.IDLE:
                self.needs_homing = False
                driver = config.machine.driver
                task_mgr.add_coroutine(lambda ctx: driver.home())
        self._update_actions_and_ui()

    def _on_connection_status_changed(
        self,
        machine: Machine,
        status: TransportStatus,
        message: Optional[str] = None,
    ):
        """Called when the active machine's connection status changes."""
        if (
            status == TransportStatus.CONNECTED
            and machine.clear_alarm_on_connect
            and machine.device_state.status == DeviceStatus.ALARM
        ):
            logger.info(
                "Machine connected in ALARM state. Auto-clearing alarm."
            )
            self.machine_cmd.clear_alarm(machine)
        self._update_actions_and_ui()

    def _on_machine_hours_changed(self, sender, **kwargs):
        """
        Called when machine hours change. Checks for maintenance notifications.
        """
        due_counters = sender.consume_due_notifications()
        for counter in due_counters:
            msg = _(
                "Maintenance Alert: {name} has reached its limit "
                "({curr:g}h / {limit:g}h)"
            ).format(
                name=counter.name,
                curr=counter.value,
                limit=counter.notify_at,
            )
            self._on_editor_notification(self, msg, persistent=True)

    def on_history_changed(
        self, history_manager: HistoryManager, command: Command
    ):
        self._update_actions_and_ui()
        # After undo/redo, the document state may have changed in ways
        # that require a full UI sync (e.g., layer visibility).
        self.on_doc_changed(self.doc_editor.doc)
        self._update_macros_menu()

    def on_doc_changed(self, sender, **kwargs):
        # Synchronize UI elements that depend on the document model
        self.surface.update_from_doc()
        doc = self.doc_editor.doc
        if doc.active_layer and doc.active_layer.workflow:
            self.workflowview.set_workflow(doc.active_layer.workflow)

        # Sync the selectability of stock items based on active layer
        self._sync_element_selectability()

        # Update button sensitivity and other state
        self._update_actions_and_ui()

    def _sync_element_selectability(self):
        """
        Updates the 'selectable' property of StockElements on the canvas
        based on which layer is currently active and their visibility.
        """
        # Find all StockElement instances currently on the canvas
        for element in self.surface.find_by_type(StockElement):
            # Stock items are only selectable when they are visible
            element.selectable = element.visible

    def _on_active_layer_changed(self, sender):
        """
        Handles activation of a new layer. Updates the workflow view and
        resets the paste counter.
        """
        logger.debug("Active layer changed, updating UI.")
        # Reset the paste counter to ensure the next paste is in-place.
        self.doc_editor.edit.reset_paste_counter()

        # Get the newly activated layer from the document
        activated_layer = self.doc_editor.doc.active_layer
        has_workflow = activated_layer.workflow is not None

        # Show/hide the workflow view based on the layer type
        self.workflowview.set_visible(has_workflow)

        if has_workflow:
            # For regular layers, update the workflow view with the
            # new workflow
            self.workflowview.set_workflow(activated_layer.workflow)

    def _on_editor_notification(
        self,
        sender,
        message: str,
        persistent: bool = False,
        action_label: Optional[str] = None,
        action_callback: Optional[Callable] = None,
    ):
        """
        Shows a toast when requested by the DocEditor.
        If 'persistent' is True, the toast will have a dismiss button and
        remain visible until closed.
        If 'action_label' and 'action_callback' are provided, a button
        will be added to the toast that triggers the callback.
        """
        toast = Adw.Toast.new(message)
        if persistent:
            toast.set_timeout(0)  # 0 = persistent
            toast.set_priority(Adw.ToastPriority.HIGH)

        if action_label and action_callback:
            toast.set_button_label(action_label)
            # Connecting directly to 'button-clicked' is the simplest way
            # to handle a callback without defining a GAction.
            toast.connect("button-clicked", lambda t: action_callback())

        self._add_toast(toast)

    def _add_toast(self, toast: Adw.Toast):
        """Helper to add a toast to the overlay and track it."""
        self._active_toasts.append(toast)
        # Connect to dismissed signal to clean up our reference
        toast.connect("dismissed", self._on_toast_dismissed)
        self.toast_overlay.add_toast(toast)

    def _on_toast_dismissed(self, toast):
        """Removes the toast from the tracking list when dismissed."""
        if toast in self._active_toasts:
            self._active_toasts.remove(toast)

    def _on_surface_transform_end(self, sender, *args, **kwargs):
        """Clears all active toasts from the toast overlay."""
        logger.debug("Clearing all toasts from overlay.")

        # Iterate over a copy of the list because dismiss() triggers removal
        for toast in list(self._active_toasts):
            toast.dismiss()

    def _on_assembly_for_preview_finished(
        self,
        handle: Optional[JobArtifactHandle],
        error: Optional[Exception],
    ):
        """Callback for when the job assembly for previews is complete."""
        if error:
            logger.error(
                "Failed to aggregate ops for preview/simulation",
                exc_info=error,
            )
            if handle:
                get_context().artifact_store.release(handle)
            handle = None

        # Schedule the UI update on the main thread, passing the handle.
        # The handle will be released in the main thread callback.
        GLib.idle_add(self._on_previews_ready, handle)

    def _on_previews_ready(self, handle: Optional[JobArtifactHandle]):
        """
        Main-thread callback to distribute assembled Ops to all consumers.
        This method is responsible for releasing the artifact handle.
        """
        final_artifact = None
        artifact_store = get_context().artifact_store
        try:
            if handle:
                final_artifact = artifact_store.get(handle)
                assert isinstance(final_artifact, JobArtifact)

            # 1. Update Simulation
            self.simulator_cmd.reload_simulation(final_artifact)

            # 2. Update G-code Preview
            gcode_action = self.action_manager.get_action(
                "toggle_gcode_preview"
            )
            state = gcode_action.get_state()
            is_gcode_visible = state and state.get_boolean()

            if is_gcode_visible and final_artifact:
                self._update_gcode_preview(
                    final_artifact.machine_code, final_artifact.op_map
                )
            else:
                self._update_gcode_preview(None, None)

        finally:
            if handle:
                artifact_store.release(handle)

        return GLib.SOURCE_REMOVE

    def refresh_previews(self):
        """
        Public method to trigger a refresh of all data previews, like the
        simulator and G-code view.
        """
        is_sim_active = self.simulator_cmd.simulation_overlay is not None
        gcode_action = self.action_manager.get_action("toggle_gcode_preview")
        gcode_state = gcode_action.get_state() if gcode_action else None
        is_gcode_visible = gcode_state and gcode_state.get_boolean()

        if not is_sim_active and not is_gcode_visible:
            return

        config = get_context().config
        if not config.machine:
            # Pass None to clear previews if no machine is configured
            self._on_previews_ready(None)
            return

        self.doc_editor.file.assemble_job_in_background(
            when_done=self._on_assembly_for_preview_finished
        )

    def _on_document_settled(self, sender):
        """
        Called when all background processing is complete. This is the main
        hook for refreshing previews that depend on the final assembled job.
        """
        self.refresh_previews()

    def _on_selection_changed(
        self,
        sender,
        elements: List[CanvasElement],
        active_element: Optional[CanvasElement],
    ):
        """Handles the 'selection-changed' signal from the WorkSurface."""
        # Get all selected DocItems (WorkPieces, Groups, etc.)
        selected_items = [
            elem.data for elem in elements if isinstance(elem.data, DocItem)
        ]

        # Get the primary active item from the signal payload
        active_item = (
            active_element.data
            if active_element and isinstance(active_element.data, DocItem)
            else None
        )

        # Reorder the list to put the active one first, if it exists
        if active_item and active_item in selected_items:
            selected_items.remove(active_item)
            selected_items.insert(0, active_item)

        self.item_props_widget.set_items(selected_items)
        self.item_revealer.set_reveal_child(bool(selected_items))
        self.sketch_props_widget.set_items(selected_items)
        self._update_actions_and_ui()

    def on_config_changed(self, sender, **kwargs):
        # Disconnect from the previously active machine, if any
        if self._current_machine:
            self._current_machine.state_changed.disconnect(
                self._on_machine_status_changed
            )
            self._current_machine.connection_status_changed.disconnect(
                self._on_connection_status_changed
            )
            self._current_machine.job_finished.disconnect(
                self._on_job_finished
            )
            self._current_machine.changed.disconnect(self._update_macros_menu)
            self._current_machine.machine_hours.changed.disconnect(
                self._on_machine_hours_changed
            )

        config = get_context().config
        self._current_machine = config.machine

        # Connect to the new active machine's signals
        if self._current_machine:
            self._current_machine.state_changed.connect(
                self._on_machine_status_changed
            )
            self._current_machine.connection_status_changed.connect(
                self._on_connection_status_changed
            )
            self._current_machine.job_finished.connect(self._on_job_finished)
            self._current_machine.changed.connect(self._update_macros_menu)
            self._current_machine.machine_hours.changed.connect(
                self._on_machine_hours_changed
            )

        # Define new machine dimensions
        new_machine = config.machine
        if new_machine:
            width_mm, height_mm = new_machine.dimensions
            y_down = new_machine.y_axis_down
            x_right = new_machine.x_axis_right
            reverse_x = new_machine.reverse_x_axis
            reverse_y = new_machine.reverse_y_axis
        else:
            width_mm, height_mm = 100.0, 100.0
            y_down, x_right, reverse_x, reverse_y = (
                False,
                False,
                False,
                False,
            )

        # Update the 3D canvas to match the new machine.
        if canvas3d_initialized and hasattr(self, "view_stack"):
            # Always switch back to 2D view on machine change for simplicity.
            if self.view_stack.get_visible_child_name() == "3d":
                self.view_stack.set_visible_child_name("2d")
                action = self.action_manager.get_action("show_3d_view")
                state = action.get_state()
                if state and state.get_boolean():
                    action.set_state(GLib.Variant.new_boolean(False))

            # Replace the 3D canvas with one configured for the new machine.
            self.view_stack.remove(self.canvas3d)
            self.canvas3d = Canvas3D(
                get_context(),
                self.doc_editor.doc,
                self.doc_editor.pipeline,
                width_mm=width_mm,
                depth_mm=height_mm,
                y_down=y_down,
                x_right=x_right,
                x_negative=reverse_x,
                y_negative=reverse_y,
            )
            self.view_stack.add_named(self.canvas3d, "3d")

        # Update the status monitor to observe the new machine
        self.status_monitor.set_machine(config.machine)

        # Update the main WorkSurface AND the SketchStudio to use the new size
        self.surface.set_machine(config.machine)
        self.sketch_studio.set_world_size(width_mm, height_mm)

        self.surface.update_from_doc()
        self._update_macros_menu()
        self._update_actions_and_ui()

        # Update theme
        self.apply_theme()

        # Check for any pending notifications from the new machine immediately
        if self._current_machine:
            self._on_machine_hours_changed(self._current_machine.machine_hours)

    def apply_theme(self):
        """Reads the theme from config and applies it to the UI."""
        style_manager = Adw.StyleManager.get_default()
        config = get_context().config
        if config.theme == "light":
            style_manager.set_color_scheme(Adw.ColorScheme.FORCE_LIGHT)
        elif config.theme == "dark":
            style_manager.set_color_scheme(Adw.ColorScheme.FORCE_DARK)
        else:  # "system" or any other invalid value
            style_manager.set_color_scheme(Adw.ColorScheme.DEFAULT)

    def on_running_tasks_changed(self, sender, tasks, progress):
        self._update_actions_and_ui()

    def _update_actions_and_ui(self):
        config = get_context().config
        active_machine = config.machine
        am = self.action_manager
        doc = self.doc_editor.doc

        if not active_machine:
            am.get_action("export").set_enabled(False)
            am.get_action("machine-settings").set_enabled(False)
            am.get_action("machine-home").set_enabled(False)
            am.get_action("machine-frame").set_enabled(False)
            am.get_action("machine-send").set_enabled(False)
            am.get_action("machine-hold").set_enabled(False)
            am.get_action("machine-cancel").set_enabled(False)
            am.get_action("machine-clear-alarm").set_enabled(False)
            am.get_action("execute-macro").set_enabled(False)
            self.toolbar.export_button.set_tooltip_text(
                _("Select a machine to enable G-code export")
            )
            self.toolbar.machine_warning_box.set_visible(False)
            self.surface.set_laser_dot_visible(False)
        else:
            device_status = active_machine.device_state.status
            conn_status = active_machine.connection_status
            state = active_machine.device_state
            active_driver = active_machine.driver

            can_export = doc.has_result() and not task_mgr.has_tasks()
            am.get_action("export").set_enabled(can_export)
            export_tooltip = _("Generate G-code")
            if task_mgr.has_tasks():
                export_tooltip = _(
                    "Cannot export while other tasks are running"
                )
            elif not doc.has_workpiece():
                export_tooltip = _("Add a workpiece to enable export")
            elif not doc.has_result():
                export_tooltip = _(
                    "Add or enable a processing step to enable export"
                )
            self.toolbar.export_button.set_tooltip_text(export_tooltip)

            self.toolbar.machine_warning_box.set_visible(
                bool(active_driver and active_driver.setup_error)
            )
            am.get_action("machine-settings").set_enabled(True)

            # A job/task is running if the machine is not idle or a UI task is
            # active.
            is_job_or_task_active = (
                device_status != DeviceStatus.IDLE or task_mgr.has_tasks()
            )

            am.get_action("machine-home").set_enabled(
                not is_job_or_task_active
            )

            can_frame = (
                active_machine.can_frame()
                and doc.has_result()
                and not is_job_or_task_active
            )
            am.get_action("machine-frame").set_enabled(can_frame)
            if not active_machine.can_frame():
                self.toolbar.frame_button.set_tooltip_text(
                    _("Configure frame power to enable")
                )
            else:
                self.toolbar.frame_button.set_tooltip_text(
                    _("Cycle laser head around the occupied area")
                )

            send_sensitive = (
                not isinstance(active_driver, NoDeviceDriver)
                and (active_driver and not active_driver.setup_error)
                and conn_status == TransportStatus.CONNECTED
                and doc.has_result()
                and not is_job_or_task_active
            )
            am.get_action("machine-send").set_enabled(send_sensitive)
            self.toolbar.send_button.set_tooltip_text(_("Send to machine"))

            hold_sensitive = device_status in (
                DeviceStatus.RUN,
                DeviceStatus.HOLD,
                DeviceStatus.CYCLE,
            )
            is_holding = device_status == DeviceStatus.HOLD
            am.get_action("machine-hold").set_enabled(hold_sensitive)
            am.get_action("machine-hold").set_state(
                GLib.Variant.new_boolean(is_holding)
            )
            if is_holding:
                self.toolbar.hold_button.set_child(self.toolbar.hold_on_icon)
                self.toolbar.hold_button.set_tooltip_text(_("Resume machine"))
            else:
                self.toolbar.hold_button.set_child(self.toolbar.hold_off_icon)
                self.toolbar.hold_button.set_tooltip_text(_("Pause machine"))

            cancel_sensitive = device_status in (
                DeviceStatus.RUN,
                DeviceStatus.HOLD,
                DeviceStatus.JOG,
                DeviceStatus.CYCLE,
            )
            am.get_action("machine-cancel").set_enabled(cancel_sensitive)

            clear_alarm_sensitive = device_status == DeviceStatus.ALARM
            am.get_action("machine-clear-alarm").set_enabled(
                clear_alarm_sensitive
            )
            if clear_alarm_sensitive:
                self.toolbar.clear_alarm_button.add_css_class(
                    "suggested-action"
                )
            else:
                self.toolbar.clear_alarm_button.remove_css_class(
                    "suggested-action"
                )

            # Update focus button sensitivity
            head = active_machine.get_default_head()
            can_focus = (
                head
                and head.focus_power_percent > 0
                and not is_job_or_task_active
            )
            am.get_action("toggle-focus").set_enabled(can_focus)

            connected = conn_status == TransportStatus.CONNECTED
            self.surface.set_laser_dot_visible(connected)
            if state and connected:
                x, y = state.machine_pos[:2]
                if x is not None and y is not None:
                    self.surface.set_laser_dot_position(x, y)

            # Set macro action sensitivity
            can_run_macros = connected and not is_job_or_task_active
            am.get_action("execute-macro").set_enabled(can_run_macros)

        # Update actions that don't depend on the machine state
        selected_elements = self.surface.get_selected_elements()
        has_selection = len(selected_elements) > 0

        am.get_action("undo").set_enabled(
            self.doc_editor.history_manager.can_undo()
        )
        am.get_action("redo").set_enabled(
            self.doc_editor.history_manager.can_redo()
        )
        am.get_action("cut").set_enabled(has_selection)
        am.get_action("copy").set_enabled(has_selection)
        am.get_action("paste").set_enabled(self.doc_editor.edit.can_paste())
        am.get_action("select_all").set_enabled(doc.has_workpiece())
        am.get_action("duplicate").set_enabled(has_selection)
        am.get_action("remove").set_enabled(has_selection)
        am.get_action("clear").set_enabled(doc.has_workpiece())

        # Update sensitivity for Grouping actions
        can_group = len(selected_elements) >= 2
        am.get_action("group").set_enabled(can_group)

        can_ungroup = any(
            isinstance(elem.data, Group) for elem in selected_elements
        )
        am.get_action("ungroup").set_enabled(can_ungroup)

        # Update sensitivity for Layer actions
        can_move_layers = has_selection and len(doc.layers) > 1
        am.get_action("layer-move-up").set_enabled(can_move_layers)
        am.get_action("layer-move-down").set_enabled(can_move_layers)

        # Update sensitivity for 3D view actions
        is_3d_view_active = self.view_stack.get_visible_child_name() == "3d"
        can_show_3d = canvas3d_initialized and not task_mgr.has_tasks()
        am.get_action("show_3d_view").set_enabled(can_show_3d)
        am.get_action("view_top").set_enabled(is_3d_view_active)
        am.get_action("view_front").set_enabled(is_3d_view_active)
        am.get_action("view_iso").set_enabled(is_3d_view_active)
        am.get_action("view_toggle_perspective").set_enabled(is_3d_view_active)

        # Update sensitivity for Arrangement actions
        can_distribute = len(self.surface.get_selected_workpieces()) >= 2
        am.get_action("align-h-center").set_enabled(has_selection)
        am.get_action("align-v-center").set_enabled(has_selection)
        am.get_action("align-left").set_enabled(has_selection)
        am.get_action("align-right").set_enabled(has_selection)
        am.get_action("align-top").set_enabled(has_selection)
        am.get_action("align-bottom").set_enabled(has_selection)
        am.get_action("spread-h").set_enabled(can_distribute)
        am.get_action("spread-v").set_enabled(can_distribute)
        self.toolbar.arrange_menu_button.set_sensitive(has_selection)

        # Update sensitivity for Tab buttons
        show_tabs_action = am.get_action("show_tabs")
        has_any_tabs = any(wp.tabs for wp in doc.all_workpieces)
        show_tabs_action.set_enabled(has_any_tabs)

        # Layout - Update sensitivity for the pixel-perfect layout action
        has_workpieces = len(doc.active_layer.get_descendants(WorkPiece)) > 0
        am.get_action("layout-pixel-perfect").set_enabled(has_workpieces)

    def on_machine_warning_clicked(self, sender):
        """Opens the machine settings dialog for the current machine."""
        config = get_context().config
        if not config.machine:
            return
        dialog = MachineSettingsDialog(
            machine=config.machine,
            transient_for=self,
        )
        dialog.present()

    def on_status_bar_clicked(self, sender):
        config = get_context().config
        dialog = MachineLogDialog(self, config.machine)
        dialog.notification_requested.connect(self._on_dialog_notification)
        dialog.present(self)

    def _on_dialog_notification(self, sender, message: str = ""):
        """Shows a toast when requested by a child dialog."""
        toast = Adw.Toast.new(message)
        self._add_toast(toast)

    def on_quit_action(self, action, parameter):
        self.close()

    def on_menu_import(self, action, param=None):
        import_handler.start_interactive_import(self, self.doc_editor)

    def on_open_clicked(self, sender):
        self.on_menu_import(sender)

    def on_clear_clicked(self, action, param):
        self.doc_editor.edit.clear_all_items()

    def on_export_clicked(self, action, param=None):
        file_dialogs.show_export_gcode_dialog(
            self, self._on_save_dialog_response
        )

    def _on_save_dialog_response(self, dialog, result, user_data):
        try:
            file = dialog.save_finish(result)
            if not file:
                return
            file_path = Path(file.get_path())
        except GLib.Error as e:
            logger.error(f"Error saving file: {e.message}")
            return

        # This is now a non-blocking call.
        self.doc_editor.file.export_gcode_to_path(file_path)

    def on_home_clicked(self, action, param):
        config = get_context().config
        if not config.machine:
            return

        # Disable focus mode when homing
        focus_action = self.action_manager.get_action("toggle-focus")
        focus_state = focus_action.get_state()
        if focus_state and focus_state.get_boolean():
            focus_action.change_state(GLib.Variant.new_boolean(False))

        self.machine_cmd.home_machine(config.machine)

    def _run_machine_job(self, job_coroutine: Coroutine):
        """
        Wraps a machine job coroutine in an asyncio.Task and handles
        its completion or failure.
        """
        fut = asyncio.run_coroutine_threadsafe(job_coroutine, task_mgr.loop)
        # Add a callback to handle the result (or exception) of the task
        fut.add_done_callback(self._on_job_future_done)

    def on_frame_clicked(self, action, param):
        config = get_context().config
        if not config.machine:
            return

        # Disable focus mode when framing
        focus_action = self.action_manager.get_action("toggle-focus")
        focus_state = focus_action.get_state()
        if focus_state and focus_state.get_boolean():
            focus_action.change_state(GLib.Variant.new_boolean(False))

        # Get the coroutine object for the framing job
        job_coro = self.machine_cmd.frame_job(
            config.machine, on_progress=self._on_job_progress_updated
        )
        # Run the job using the helper
        self._run_machine_job(job_coro)

    def on_send_clicked(self, action, param):
        config = get_context().config
        if not config.machine:
            return

        # Disable focus mode when sending
        focus_action = self.action_manager.get_action("toggle-focus")
        focus_state = focus_action.get_state()
        if focus_state and focus_state.get_boolean():
            focus_action.change_state(GLib.Variant.new_boolean(False))

        # Get the coroutine object for the send job
        job_coro = self.machine_cmd.send_job(
            config.machine, on_progress=self._on_job_progress_updated
        )
        # Run the job using the helper
        self._run_machine_job(job_coro)

    def on_hold_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        """
        Handles the 'change-state' signal for the 'hold' action.
        This is the correct handler for a stateful action.
        """
        config = get_context().config
        if not config.machine:
            return
        is_requesting_hold = value.get_boolean()
        self.machine_cmd.set_hold(config.machine, is_requesting_hold)
        action.set_state(value)

    def on_cancel_clicked(self, action, param):
        config = get_context().config
        if not config.machine:
            return
        self.machine_cmd.cancel_job(config.machine)

    def on_clear_alarm_clicked(self, action, param):
        config = get_context().config
        if not config.machine:
            return
        self.machine_cmd.clear_alarm(config.machine)

    def on_toggle_focus_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        """
        Handles the 'change-state' signal for the 'toggle-focus' action.
        This toggles the laser focus mode on/off.
        """
        config = get_context().config
        if not config.machine:
            return

        is_focus_on = value.get_boolean()
        head = config.machine.get_default_head()

        if is_focus_on:
            self.machine_cmd.set_power(head, head.focus_power_percent)
        else:
            self.machine_cmd.set_power(head, 0)
        action.set_state(value)

        # Update the toolbar button icon
        if is_focus_on:
            self.toolbar.focus_button.set_child(self.toolbar.focus_off_icon)
        else:
            self.toolbar.focus_button.set_child(self.toolbar.focus_on_icon)

    def on_jog_clicked(self, action, param):
        """Show the jog control dialog."""
        config = get_context().config
        if not config.machine:
            return

        dialog = JogDialog(
            machine=config.machine,
            machine_cmd=self.machine_cmd,
        )
        dialog.set_transient_for(self)
        dialog.present()

    def on_elements_deleted(self, sender, elements: List[CanvasElement]):
        """Handles the deletion signal from the WorkSurface."""
        items_to_delete = [
            elem.data for elem in elements if isinstance(elem.data, DocItem)
        ]
        if items_to_delete:
            self.doc_editor.edit.remove_items(
                items_to_delete, "Delete item(s)"
            )

    def on_cut_requested(self, sender, items: List[DocItem]):
        """Handles the 'cut-requested' signal from the WorkSurface."""
        self.doc_editor.edit.cut_items(items)
        self._update_actions_and_ui()

    def on_copy_requested(self, sender, items: List[DocItem]):
        """
        Handles the 'copy-requested' signal from the WorkSurface.
        """
        self.doc_editor.edit.copy_items(items)
        self._update_actions_and_ui()

    def on_paste_requested(self, sender, *args):
        """
        Handles the 'paste-requested' signal from the WorkSurface.
        Checks for image data on system clipboard first, then falls back
        to workpiece paste.
        """
        # Priority 1: Check if system clipboard contains image data
        if self.drag_drop_cmd.handle_clipboard_paste():
            return

        # Priority 2: Standard workpiece paste
        newly_pasted = self.doc_editor.edit.paste_items()
        if newly_pasted:
            self.surface.select_items(newly_pasted)
        self._update_actions_and_ui()

    def on_select_all(self, action, param):
        """
        Selects all top-level items (workpieces and groups) in the document.
        """
        doc = self.doc_editor.doc
        items_to_select = doc.get_top_level_items()
        if items_to_select:
            self.surface.select_items(items_to_select)

    def on_duplicate_requested(self, sender, items: List[DocItem]):
        """
        Handles the 'duplicate-requested' signal from the WorkSurface.
        """
        newly_duplicated = self.doc_editor.edit.duplicate_items(items)
        if newly_duplicated:
            self.surface.select_items(newly_duplicated)

    def on_menu_cut(self, action, param):
        selection = self.surface.get_selected_items()
        if selection:
            self.doc_editor.edit.cut_items(list(selection))
            self._update_actions_and_ui()

    def on_menu_copy(self, action, param):
        selection = self.surface.get_selected_items()
        if selection:
            self.doc_editor.edit.copy_items(list(selection))
            self._update_actions_and_ui()

    def on_menu_duplicate(self, action, param):
        selection = self.surface.get_selected_items()
        if selection:
            newly_duplicated = self.doc_editor.edit.duplicate_items(
                list(selection)
            )
            self.surface.select_items(newly_duplicated)

    def on_menu_remove(self, action, param):
        items = self.surface.get_selected_items()
        if items:
            self.doc_editor.edit.remove_items(list(items))

    def show_about_dialog(self, action, param):
        dialog = AboutDialog(transient_for=self)
        dialog.present()

    def show_settings(self, action, param):
        dialog = SettingsWindow(transient_for=self)
        dialog.present()
        dialog.connect("close-request", self._on_settings_dialog_closed)

    def show_machine_settings(self, action, param):
        """Opens the machine settings dialog for the current machine."""
        config = get_context().config
        if not config.machine:
            return
        dialog = MachineSettingsDialog(
            machine=config.machine,
            transient_for=self,
        )
        dialog.present()

    def on_show_material_test(self, action, param):
        """Creates a material test grid by delegating to the editor command."""
        self.doc_editor.material_test.create_test_grid()

    def _on_settings_dialog_closed(self, dialog):
        logger.debug("Settings dialog closed")
        self.surface.grab_focus()  # re-enables keyboard shortcuts

    def _on_job_time_updated(self, sender, *, total_seconds):
        """
        Handles the preview_time_updated signal from the pipeline.
        Updates the status bar with the total estimated time.
        """
        self.status_monitor.set_estimated_time(total_seconds)

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handle key press events, ESC to exit simulation mode."""
        if keyval == Gdk.KEY_Escape:
            # Check if simulation mode is active
            if self.surface.is_simulation_mode():
                # Exit simulation mode
                self.simulator_cmd._exit_mode()
                return True  # Event handled
        return False  # Allow other key presses to be processed normally
