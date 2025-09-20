import logging
from pathlib import Path
from typing import List, Optional, Callable, cast
from gi.repository import Gtk, Gio, GLib, Gdk, Adw
from . import __version__
from .shared.tasker import task_mgr
from .config import config, config_mgr
from .machine.driver.driver import DeviceStatus, DeviceState
from .machine.driver.dummy import NoDeviceDriver
from .machine.models.machine import Machine
from .core.group import Group
from .core.item import DocItem
from .core.layer import Layer
from .core.stock import StockItem
from .core.stocklayer import StockLayer
from .pipeline.steps import (
    create_contour_step,
    create_outline_step,
    create_raster_step,
)
from .undo import HistoryManager, Command
from .doceditor.editor import DocEditor
from .doceditor.ui.workflow_view import WorkflowView
from .workbench.surface import WorkSurface
from .workbench.elements.stock import StockElement
from .doceditor.ui.layer_list import LayerListView
from .machine.transport import TransportStatus
from .shared.ui.task_bar import TaskBar
from .machine.ui.log_dialog import MachineLogDialog
from .shared.ui.preferences_dialog import PreferencesWindow
from .machine.ui.settings_dialog import MachineSettingsDialog
from .doceditor.ui.item_properties import DocItemPropertiesWidget
from .workbench.canvas import CanvasElement
from .shared.ui.about import AboutDialog
from .toolbar import MainToolbar
from .actions import ActionManager
from .main_menu import MainMenu
from .workbench.view_mode_cmd import ViewModeCmd
from .workbench.canvas3d import Canvas3D, initialized as canvas3d_initialized
from .doceditor.ui import file_dialogs, import_handler


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

        # The ToastOverlay will wrap the main content box
        self.toast_overlay = Adw.ToastOverlay()
        self.set_content(self.toast_overlay)

        # The main content box is now the child of the ToastOverlay
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.toast_overlay.set_child(vbox)

        # Create the central document editor. This now owns the Doc and
        # OpsGenerator.
        assert config_mgr is not None
        self.doc_editor = DocEditor(task_mgr, config_mgr)

        # Instantiate UI-specific command handlers
        self.view_cmd = ViewModeCmd(self.doc_editor)

        # Add a global click handler to manage focus correctly.
        root_click_gesture = Gtk.GestureClick.new()
        root_click_gesture.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        root_click_gesture.connect("pressed", self._on_root_click_pressed)
        self.add_controller(root_click_gesture)

        geometry = _get_monitor_geometry()
        if geometry:
            self.set_default_size(
                int(geometry.width * 0.8), int(geometry.height * 0.8)
            )
        else:
            self.set_default_size(1100, 800)

        # HeaderBar with left-aligned menu and centered title
        header_bar = Adw.HeaderBar()
        vbox.append(header_bar)

        # Create the menu model and the popover menubar
        menu_model = MainMenu()
        menubar = Gtk.PopoverMenuBar.new_from_model(menu_model)
        menubar.add_css_class("in-header-menubar")
        header_bar.pack_start(menubar)

        # Create and set the centered title widget
        window_title = Adw.WindowTitle(
            title=self.get_title() or "", subtitle=__version__ or ""
        )
        header_bar.set_title_widget(window_title)

        # Create and add the main toolbar.
        self.toolbar = MainToolbar()
        self._connect_toolbar_signals()
        vbox.append(self.toolbar)

        # Create the Paned splitting the window into left and right sections.
        self.paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self.paned.set_vexpand(True)
        vbox.append(self.paned)

        # Apply styles
        self.paned.add_css_class("mainpaned")
        display = Gdk.Display.get_default()
        if display:
            provider = Gtk.CssProvider()
            provider.load_from_string(css)
            Gtk.StyleContext.add_provider_for_display(
                display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )

        # Determine initial machine dimensions for the 3D canvas and surface
        if config.machine:
            width_mm, height_mm = config.machine.dimensions
            y_down = getattr(config.machine, "y_axis_down", False)
        else:
            # Default to a square aspect ratio if no machine is configured
            width_mm, height_mm = 100.0, 100.0
            y_down = False

        self.surface = WorkSurface(
            editor=self.doc_editor,
            machine=config.machine,
            cam_visible=True,  # Will be set by action state
        )
        self.surface.set_hexpand(True)

        # Setup keyboard actions using the new ActionManager.
        self.action_manager = ActionManager(self)
        self.action_manager.register_actions()
        app = self.get_application()
        if app:
            self.action_manager.set_accelerators(app)

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
        doc.descendant_updated.connect(self.on_doc_changed)
        doc.descendant_transform_changed.connect(self.on_doc_changed)
        doc.active_layer_changed.connect(self._on_active_layer_changed)
        doc.history_manager.changed.connect(self.on_history_changed)

        # Connect editor signals
        self.doc_editor.notification_requested.connect(
            self._on_editor_notification
        )

        # Create the view stack for 2D and 3D views
        self.view_stack = Gtk.Stack()
        self.view_stack.set_transition_type(
            Gtk.StackTransitionType.SLIDE_LEFT_RIGHT
        )
        self.view_stack.set_margin_start(12)
        self.view_stack.set_hexpand(True)
        self.paned.set_start_child(self.view_stack)

        self.view_stack.add_named(self.surface, "2d")

        if canvas3d_initialized:
            self.canvas3d = Canvas3D(
                self.doc_editor.doc,
                width_mm=width_mm,
                depth_mm=height_mm,
                y_down=y_down,
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

        # Add the Layer list view
        self.layer_list_view = LayerListView(self.doc_editor.doc)
        self.layer_list_view.set_margin_end(12)
        right_pane_box.append(self.layer_list_view)

        # The WorkflowView will be updated when a layer is activated.
        initial_workflow = self.doc_editor.doc.active_layer.workflow
        assert initial_workflow, "Initial active layer must have a workflow"
        step_factories: List[Callable] = [
            create_contour_step,
            create_outline_step,
            create_raster_step,
        ]
        self.workflowview = WorkflowView(
            initial_workflow, step_factories=step_factories
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

        # Connect signals for item selection and actions
        self.surface.selection_changed.connect(self._on_selection_changed)
        self.surface.elements_deleted.connect(self.on_elements_deleted)
        self.surface.cut_requested.connect(self.on_cut_requested)
        self.surface.copy_requested.connect(self.on_copy_requested)
        self.surface.paste_requested.connect(self.on_paste_requested)
        self.surface.duplicate_requested.connect(self.on_duplicate_requested)

        # Create and add the status monitor widget.
        self.status_monitor = TaskBar(task_mgr)
        self.status_monitor.log_requested.connect(self.on_status_bar_clicked)
        vbox.append(self.status_monitor)

        # Set up config signals.
        config.changed.connect(self.on_config_changed)
        task_mgr.tasks_updated.connect(self.on_running_tasks_changed)
        self.needs_homing = (
            config.machine.home_on_start if config.machine else False
        )

        # Set initial state
        self.on_config_changed(None)

    def on_show_3d_view(
        self, action: Gio.SimpleAction, value: Optional[GLib.Variant]
    ):
        """Delegates the view switching logic to the command module."""
        self.view_cmd.toggle_3d_view(self, action, value)

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
        action.set_state(value)

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

        # Find the first non-stock, "workpiece" layer to add the default
        # step to.
        first_workpiece_layer = None
        for layer in doc.layers:
            if not isinstance(layer, StockLayer):
                first_workpiece_layer = layer
                break  # Found the first one

        if (
            first_workpiece_layer
            and first_workpiece_layer.workflow
            and not first_workpiece_layer.workflow.has_steps()
        ):
            workflow = first_workpiece_layer.workflow
            default_step = create_contour_step()
            workflow.add_step(default_step)
            logger.info("Added default Contour step to initial document.")

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

    def _on_root_click_pressed(self, gesture, n_press, x, y):
        """
        Global click handler to unfocus widgets when clicking on "dead space".
        """
        self.surface.grab_focus()

    def on_machine_selected_by_selector(self, sender, *, machine: Machine):
        """
        Handles the 'machine_selected' signal from the MachineSelector widget.
        The signature is compatible with the blinker library.
        """
        # The widget's signal is the source of truth for user-driven changes.
        # We just need to update the global config.
        if config.machine is None or config.machine.id != machine.id:
            logger.info(f"User selected machine via dropdown: {machine.name}")
            config.set_machine(machine)
            self.surface.set_machine(machine)

    def _on_machine_status_changed(self, machine: Machine, state: DeviceState):
        """Called when the active machine's state changes."""
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
            self.doc_editor.machine.clear_alarm(machine)
        self._update_actions_and_ui()

    def on_history_changed(
        self, history_manager: HistoryManager, command: Command
    ):
        self._update_actions_and_ui()
        # After undo/redo, the document state may have changed in ways
        # that require a full UI sync (e.g., layer visibility).
        self.on_doc_changed(self.doc_editor.doc)

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
        based on which layer is currently active.
        """
        # Find all StockElement instances currently on the canvas
        for element in self.surface.find_by_type(StockElement):
            stock_item = cast(StockItem, element.data)
            # An item is selectable if its parent layer is the active one
            layer = cast(Layer, stock_item.parent)
            is_selectable = layer and layer.active
            element.selectable = is_selectable

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

    def _on_editor_notification(self, sender, message: str):
        """Shows a toast when requested by the DocEditor."""
        self.toast_overlay.add_toast(Adw.Toast.new(message))

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

        self._current_machine = config.machine

        # Connect to the new active machine's signals
        if self._current_machine:
            self._current_machine.state_changed.connect(
                self._on_machine_status_changed
            )
            self._current_machine.connection_status_changed.connect(
                self._on_connection_status_changed
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

            new_machine = config.machine
            if new_machine:
                width_mm, height_mm = new_machine.dimensions
                y_down = getattr(new_machine, "y_axis_down", False)
            else:
                width_mm, height_mm = 100.0, 100.0
                y_down = False

            self.canvas3d = Canvas3D(
                self.doc_editor.doc,
                width_mm=width_mm,
                depth_mm=height_mm,
                y_down=y_down,
            )
            self.view_stack.add_named(self.canvas3d, "3d")

        # Update the status monitor to observe the new machine
        self.status_monitor.set_machine(config.machine)

        self.surface.update_from_doc()
        self._update_actions_and_ui()

        # Update theme
        self.apply_theme()

    def apply_theme(self):
        """Reads the theme from config and applies it to the UI."""
        style_manager = Adw.StyleManager.get_default()
        if config.theme == "light":
            style_manager.set_color_scheme(Adw.ColorScheme.FORCE_LIGHT)
        elif config.theme == "dark":
            style_manager.set_color_scheme(Adw.ColorScheme.FORCE_DARK)
        else:  # "system" or any other invalid value
            style_manager.set_color_scheme(Adw.ColorScheme.DEFAULT)

    def on_running_tasks_changed(self, sender, tasks, progress):
        self._update_actions_and_ui()

    def _update_actions_and_ui(self):
        active_machine = config.machine
        am = self.action_manager
        doc = self.doc_editor.doc

        if not active_machine:
            am.get_action("export").set_enabled(False)
            am.get_action("machine_settings").set_enabled(False)
            am.get_action("home").set_enabled(False)
            am.get_action("frame").set_enabled(False)
            am.get_action("send").set_enabled(False)
            am.get_action("hold").set_enabled(False)
            am.get_action("cancel").set_enabled(False)
            am.get_action("clear_alarm").set_enabled(False)
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

            can_export = doc.has_workpiece() and not task_mgr.has_tasks()
            am.get_action("export").set_enabled(can_export)
            export_tooltip = _("Generate G-code")
            if not doc.has_workpiece():
                export_tooltip = _("Add a workpiece to enable export")
            elif task_mgr.has_tasks():
                export_tooltip = _(
                    "Cannot export while other tasks are running"
                )
            self.toolbar.export_button.set_tooltip_text(export_tooltip)

            self.toolbar.machine_warning_box.set_visible(
                bool(active_driver and active_driver.setup_error)
            )
            am.get_action("machine_settings").set_enabled(True)

            # A job/task is running if the machine is not idle or a UI task is
            # active.
            is_job_or_task_active = (
                device_status != DeviceStatus.IDLE or task_mgr.has_tasks()
            )

            am.get_action("home").set_enabled(not is_job_or_task_active)

            can_frame = (
                active_machine.can_frame()
                and doc.has_result()
                and not is_job_or_task_active
            )
            am.get_action("frame").set_enabled(can_frame)
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
            am.get_action("send").set_enabled(send_sensitive)
            self.toolbar.send_button.set_tooltip_text(_("Send to machine"))

            hold_sensitive = device_status in (
                DeviceStatus.RUN,
                DeviceStatus.HOLD,
                DeviceStatus.CYCLE,
            )
            is_holding = device_status == DeviceStatus.HOLD
            am.get_action("hold").set_enabled(hold_sensitive)
            am.get_action("hold").set_state(
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
            am.get_action("cancel").set_enabled(cancel_sensitive)

            clear_alarm_sensitive = device_status == DeviceStatus.ALARM
            am.get_action("clear_alarm").set_enabled(clear_alarm_sensitive)
            if clear_alarm_sensitive:
                self.toolbar.clear_alarm_button.add_css_class(
                    "suggested-action"
                )
            else:
                self.toolbar.clear_alarm_button.remove_css_class(
                    "suggested-action"
                )

            connected = conn_status == TransportStatus.CONNECTED
            self.surface.set_laser_dot_visible(connected)
            if state and connected:
                x, y = state.machine_pos[:2]
                if x is not None and y is not None:
                    self.surface.set_laser_dot_position(x, y)

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
        selected_top_level_items = self.surface.get_selected_top_level_items()

        if len(selected_top_level_items) >= 2:
            # Scenario: Multiple top-level items are selected.
            can_layout = True
        elif not selected_top_level_items:
            # Scenario: Nothing selected. Action will lay out all workpieces.
            can_layout = len(doc.get_top_level_items()) >= 2
        else:  # One item selected
            can_layout = False

        am.get_action("layout-pixel-perfect").set_enabled(can_layout)

    def on_machine_warning_clicked(self, sender):
        """Opens the machine settings dialog for the current machine."""
        if not config.machine:
            return
        dialog = MachineSettingsDialog(machine=config.machine)
        dialog.present(self)

    def on_status_bar_clicked(self, sender):
        dialog = MachineLogDialog(self, config.machine)
        dialog.notification_requested.connect(self._on_dialog_notification)
        dialog.present(self)

    def _on_dialog_notification(self, sender, message: str = ""):
        """Shows a toast when requested by a child dialog."""
        self.toast_overlay.add_toast(Adw.Toast.new(message))

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

        self.doc_editor.file.export_gcode_to_path(file_path)

    def on_home_clicked(self, action, param):
        if not config.machine:
            return
        self.doc_editor.machine.home_machine(config.machine)

    def on_frame_clicked(self, action, param):
        if not config.machine:
            return
        self.doc_editor.machine.frame_job(config.machine)

    def on_send_clicked(self, action, param):
        if not config.machine:
            return
        self.doc_editor.machine.send_job(config.machine)

    def on_hold_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        """
        Handles the 'change-state' signal for the 'hold' action.
        This is the correct handler for a stateful action.
        """
        if not config.machine:
            return
        is_requesting_hold = value.get_boolean()
        self.doc_editor.machine.set_hold(config.machine, is_requesting_hold)
        action.set_state(value)

    def on_cancel_clicked(self, action, param):
        if not config.machine:
            return
        self.doc_editor.machine.cancel_job(config.machine)

    def on_clear_alarm_clicked(self, action, param):
        if not config.machine:
            return
        self.doc_editor.machine.clear_alarm(config.machine)

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
        """
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

    def show_preferences(self, action, param):
        dialog = PreferencesWindow(transient_for=self)
        dialog.present()
        dialog.connect("close-request", self._on_preferences_dialog_closed)

    def show_machine_settings(self, action, param):
        """Opens the machine settings dialog for the current machine."""
        if not config.machine:
            return
        dialog = MachineSettingsDialog(machine=config.machine)
        dialog.present(self)

    def _on_preferences_dialog_closed(self, dialog):
        logger.debug("Preferences dialog closed")
        self.surface.grab_focus()  # re-enables keyboard shortcuts
