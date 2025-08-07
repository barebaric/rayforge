import asyncio
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable
from gi.repository import Gtk, Gio, GLib, Gdk, Adw  # type: ignore
from . import __version__
from .shared.tasker import task_mgr
from .shared.tasker.context import ExecutionContext
from .config import config
from .machine.driver.driver import DeviceStatus, DeviceState
from .machine.driver.dummy import NoDeviceDriver
from .machine.models.machine import Machine
from .core.doc import Doc
from .core.workpiece import WorkPiece
from .pipeline.steps import (
    create_contour_step,
    create_outline_step,
    create_raster_step,
)
from .pipeline.job import generate_job_ops
from .pipeline.encoder.gcode import GcodeEncoder
from .importer import renderers, renderer_by_mime_type, renderer_by_extension
from .undo import HistoryManager, Command, ListItemCommand, ReorderListCommand
from .doceditor.ui.workflow_view import WorkflowView
from .workbench.surface import WorkSurface
from .doceditor.ui.layer_list import LayerListView
from .machine.transport import TransportStatus
from .shared.ui.task_bar import TaskBar
from .machine.ui.log_dialog import MachineLogDialog
from .shared.ui.preferences_dialog import PreferencesWindow
from .machine.ui.settings_dialog import MachineSettingsDialog
from .doceditor.ui.workpiece_properties import WorkpiecePropertiesWidget
from .workbench.canvas import CanvasElement
from .shared.ui.about import AboutDialog
from .toolbar import MainToolbar
from .actions import ActionManager
from .main_menu import MainMenu


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


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_title(_("Rayforge"))
        self._current_machine: Optional[Machine] = None  # For signal handling

        # The main content box
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        # The ToastOverlay will wrap the main content box
        self.toast_overlay = Adw.ToastOverlay()
        self.toast_overlay.set_child(vbox)

        # Set the ToastOverlay as the window's content
        self.set_content(self.toast_overlay)

        # Add a global click handler to manage focus correctly.
        root_click_gesture = Gtk.GestureClick.new()
        root_click_gesture.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        root_click_gesture.connect("pressed", self._on_root_click_pressed)
        self.add_controller(root_click_gesture)

        display = Gdk.Display.get_default()
        monitors = display.get_monitors()

        # Try to get the monitor under the cursor (heuristic for active
        # monitor). Note: Wayland has no concept of "primary monitor"
        # anymore, so Gdk.get_primary_monitor() is obsolete.
        monitor = None
        if monitors:
            seat = display.get_default_seat()
            if seat:
                pointer = seat.get_pointer()
                if pointer:
                    surface, x, y = pointer.get_surface_at_position()
                    if surface:
                        monitor = display.get_monitor_at_surface(surface)

        # Fallback to the first monitor if no monitor is found under the cursor
        if not monitor and monitors:
            monitor = monitors[0]

        # Set the window size based on the monitor's geometry or a default size
        if monitor:
            geometry = monitor.get_geometry()
            self.set_default_size(
                int(geometry.width * 0.8), int(geometry.height * 0.8)
            )
        else:
            self.set_default_size(1100, 800)

        # Make a default document. Must be created before ActionManager.
        self.doc = Doc()

        # Setup keyboard actions using the new ActionManager.
        self.action_manager = ActionManager(self)
        self.action_manager.register_actions()
        self.action_manager.set_accelerators(self.get_application())

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
            title=self.get_title(), subtitle=__version__
        )
        header_bar.set_title_widget(window_title)

        # Stores a snapshot (list of dicts) of the copied workpieces.
        self._clipboard_snapshot: List[Dict] = []
        # Tracks the number of pastes for the current clipboard snapshot.
        self._paste_counter = 0
        # The (x, -y) offset to apply for each paste level.
        self._paste_increment_mm: Tuple[float, float] = (10.0, -10.0)

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
        provider = Gtk.CssProvider()
        provider.load_from_string(css)
        Gtk.StyleContext.add_provider_for_display(
            display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # Create a work area to display the image and paths
        if config.machine:
            width_mm, height_mm = config.machine.dimensions
            ratio = width_mm / height_mm if height_mm > 0 else 1.0
        else:
            # Default to a square aspect ratio if no machine is configured
            ratio = 1.0
        self.frame = Gtk.AspectFrame(ratio=ratio, obey_child=False)
        self.frame.set_margin_start(12)
        self.frame.set_hexpand(True)
        self.paned.set_start_child(self.frame)

        # Connect document signals
        self._initialize_document()
        self.doc.changed.connect(self.on_doc_changed)
        self.doc.active_layer_changed.connect(self._on_active_layer_changed)
        self.doc.history_manager.changed.connect(self.on_history_changed)

        self.surface = WorkSurface(
            self.doc,
            config.machine,
            cam_visible=self.toolbar.camera_visibility_button.get_active(),
        )
        self.surface.set_hexpand(True)
        self.frame.set_child(self.surface)

        # Undo/Redo buttons are now connected to the doc via actions.
        self.toolbar.undo_button.set_history_manager(self.doc.history_manager)
        self.toolbar.redo_button.set_history_manager(self.doc.history_manager)

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
        right_pane_scrolled_window.set_child(right_pane_box)

        # Add the Layer list view
        self.layer_list_view = LayerListView(self.doc)
        self.layer_list_view.set_margin_end(12)
        right_pane_box.append(self.layer_list_view)

        # The WorkflowView will be updated when a layer is activated.
        initial_workflow = self.doc.layers[0].workflow
        step_factories: List[Callable] = [
            create_contour_step,
            create_outline_step,
            create_raster_step,
        ]
        self.workflowview = WorkflowView(
            initial_workflow, step_factories=step_factories
        )
        self.workflowview.set_size_request(400, -1)
        self.workflowview.set_margin_top(20)
        self.workflowview.set_margin_end(12)
        right_pane_box.append(self.workflowview)

        # Add the WorkpiecePropertiesWidget
        self.workpiece_props_widget = WorkpiecePropertiesWidget()
        workpiece_props_container = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL
        )
        self.workpiece_props_widget.set_margin_top(20)
        self.workpiece_props_widget.set_margin_end(12)
        workpiece_props_container.append(self.workpiece_props_widget)

        self.workpiece_revealer = Gtk.Revealer()
        self.workpiece_revealer.set_child(workpiece_props_container)
        self.workpiece_revealer.set_reveal_child(False)
        self.workpiece_revealer.set_transition_type(
            Gtk.RevealerTransitionType.SLIDE_UP
        )
        right_pane_box.append(self.workpiece_revealer)

        # Connect signals for workpiece selection
        self.surface.selection_changed.connect(self._on_selection_changed)

        # Connect signals for clipboard and duplication
        self.surface.cut_requested.connect(self.on_cut_requested)
        self.surface.copy_requested.connect(self.on_copy_requested)
        self.surface.paste_requested.connect(self.on_paste_requested)
        self.surface.duplicate_requested.connect(self.on_duplicate_requested)
        self.surface.aspect_ratio_changed.connect(
            self._on_surface_aspect_changed
        )

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

    def _initialize_document(self):
        """
        Adds required initial state to a new document, such as a default
        step.
        """
        if not self.doc.layers:
            return

        first_layer = self.doc.layers[0]
        if not first_layer.workflow.has_steps():
            workflow = first_layer.workflow
            default_step = create_contour_step(workflow=workflow)
            workflow.add_step(default_step)
            logger.info("Added default Contour step to initial document.")

    def _connect_toolbar_signals(self):
        """Connects signals from the MainToolbar to their handlers.
        Most buttons are connected via Gio.Actions. Only view-state toggles
        and special widgets are connected here.
        """
        self.toolbar.visibility_toggled.connect(
            self.on_button_visibility_toggled
        )
        self.toolbar.camera_visibility_toggled.connect(
            self.on_camera_image_visibility_toggled
        )
        self.toolbar.show_travel_toggled.connect(self.on_show_travel_toggled)

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
        self.update_state()

    def _on_connection_status_changed(
        self, machine: Machine, status: TransportStatus, message: str
    ):
        """Called when the active machine's connection status changes."""
        self.update_state()

    def on_history_changed(
        self, history_manager: HistoryManager, command: Command
    ):
        self.update_state()
        # After undo/redo, the document state may have changed in ways
        # that require a full UI sync (e.g., layer visibility).
        self.on_doc_changed(self.doc)

    def on_doc_changed(self, sender, **kwargs):
        # Synchronize UI elements that depend on the document model
        self.surface.update_from_doc(self.doc)
        if self.doc.active_layer:
            self.workflowview.set_workflow(self.doc.active_layer.workflow)

        # Update button sensitivity and other state
        self.update_state()

    def _on_active_layer_changed(self, sender):
        """Resets the paste counter when the active layer changes."""
        self._paste_counter = 0
        logger.debug("Active layer changed, paste counter reset.")

    def _on_selection_changed(
        self,
        sender,
        elements: List[CanvasElement],
        active_element: Optional[CanvasElement],
    ):
        """Handles the 'selection-changed' signal from the WorkSurface."""
        # Get all selected workpieces
        selected_workpieces = [
            elem.data for elem in elements if isinstance(elem.data, WorkPiece)
        ]

        # Get the primary active workpiece from the signal payload
        active_workpiece = (
            active_element.data
            if active_element and isinstance(active_element.data, WorkPiece)
            else None
        )

        # Reorder the list to put the active one first, if it exists
        if active_workpiece and active_workpiece in selected_workpieces:
            selected_workpieces.remove(active_workpiece)
            selected_workpieces.insert(0, active_workpiece)

        self.workpiece_props_widget.set_workpieces(selected_workpieces)
        self.workpiece_revealer.set_reveal_child(bool(selected_workpieces))
        self.update_state()

    def _on_surface_aspect_changed(self, sender, ratio):
        self.frame.set_ratio(ratio)

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

        # Update the status monitor to observe the new machine
        self.status_monitor.set_machine(config.machine)

        self.surface.update_from_doc(self.doc)
        self.update_state()

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
        self.update_state()

    def update_state(self):
        active_machine = config.machine
        am = self.action_manager

        if not active_machine:
            am.get_action("export").set_enabled(False)
            am.get_action("machine_settings").set_enabled(False)
            am.get_action("home").set_enabled(False)
            am.get_action("frame").set_enabled(False)
            am.get_action("send").set_enabled(False)
            am.get_action("hold").set_enabled(False)
            am.get_action("cancel").set_enabled(False)
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

            can_export = self.doc.has_workpiece() and not task_mgr.has_tasks()
            am.get_action("export").set_enabled(can_export)
            export_tooltip = _("Generate G-code")
            if not self.doc.has_workpiece():
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
                and self.doc.has_result()
                and not is_job_or_task_active
            )
            am.get_action("frame").set_enabled(can_frame)

            send_sensitive = (
                not isinstance(active_driver, NoDeviceDriver)
                and (active_driver and not active_driver.setup_error)
                and conn_status == TransportStatus.CONNECTED
                and self.doc.has_result()
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

            connected = conn_status == TransportStatus.CONNECTED
            self.surface.set_laser_dot_visible(connected)
            if state and connected:
                x, y = state.machine_pos[:2]
                if x is not None and y is not None:
                    self.surface.set_laser_dot_position(x, y)

        # Update actions that don't depend on the machine state
        has_selection = len(self.surface.get_selected_workpieces()) > 0
        has_tasks = task_mgr.has_tasks()
        can_edit = not has_tasks

        am.get_action("undo").set_enabled(self.doc.history_manager.can_undo())
        am.get_action("redo").set_enabled(self.doc.history_manager.can_redo())
        am.get_action("cut").set_enabled(has_selection and can_edit)
        am.get_action("copy").set_enabled(has_selection)
        am.get_action("paste").set_enabled(
            len(self._clipboard_snapshot) > 0 and can_edit
        )
        am.get_action("duplicate").set_enabled(has_selection and can_edit)
        am.get_action("remove").set_enabled(has_selection and can_edit)
        am.get_action("clear").set_enabled(
            self.doc.has_workpiece() and can_edit
        )

        # Update sensitivity for all alignment buttons
        align_sensitive = has_selection and can_edit
        am.get_action("align-h-center").set_enabled(align_sensitive)
        am.get_action("align-v-center").set_enabled(align_sensitive)
        self.toolbar.align_menu_button.set_sensitive(align_sensitive)
        self.toolbar.distribute_menu_button.set_sensitive(align_sensitive)

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
        self.on_open_clicked(self)

    def on_open_clicked(self, sender):
        # Create a file chooser dialog
        dialog = Gtk.FileDialog.new()
        dialog.set_title(_("Open File"))

        # Create a Gio.ListModel for the filters
        filter_list = Gio.ListStore.new(Gtk.FileFilter)
        all_supported = Gtk.FileFilter()
        all_supported.set_name(_("All supported"))
        for renderer in renderers:
            file_filter = Gtk.FileFilter()
            file_filter.set_name(_(renderer.label))
            for mime_type in renderer.mime_types:
                file_filter.add_mime_type(mime_type)
                all_supported.add_mime_type(mime_type)
            filter_list.append(file_filter)
        filter_list.append(all_supported)

        # Set the filters for the dialog
        dialog.set_filters(filter_list)
        dialog.set_default_filter(all_supported)

        # Show the dialog and handle the response
        dialog.open(self, None, self.on_file_dialog_response)

    def on_button_visibility_toggled(self, sender, active):
        self.surface.set_workpieces_visible(active)
        if active:
            self.toolbar.visibility_button.set_child(
                self.toolbar.visibility_on_icon
            )
        else:
            self.toolbar.visibility_button.set_child(
                self.toolbar.visibility_off_icon
            )

    def on_camera_image_visibility_toggled(self, sender, active):
        self.surface.set_camera_image_visibility(active)
        if active:
            self.toolbar.camera_visibility_button.set_child(
                self.toolbar.camera_visibility_on_icon
            )
        else:
            self.toolbar.camera_visibility_button.set_child(
                self.toolbar.camera_visibility_off_icon
            )

    def on_show_travel_toggled(self, sender, active):
        self.surface.set_show_travel_moves(active)

    def on_clear_clicked(self, action, param):
        if not self.doc.workpieces:
            return

        history = self.doc.history_manager
        with history.transaction(_("Remove all workpieces")) as t:
            for layer in self.doc.layers:
                if layer.workpieces:
                    command = ReorderListCommand(
                        target_obj=layer,
                        list_property_name="workpieces",
                        new_list=[],
                        setter_method_name="set_workpieces",
                        name=_("Clear Layer Workpieces"),
                    )
                    t.execute(command)

    def on_export_clicked(self, action, param=None):
        # Create a file chooser dialog for saving the file
        dialog = Gtk.FileDialog.new()
        dialog.set_title(_("Save G-code File"))

        # Set the default file name
        dialog.set_initial_name("output.gcode")

        # Create a Gio.ListModel for the filters
        filter_list = Gio.ListStore.new(Gtk.FileFilter)
        gcode_filter = Gtk.FileFilter()
        gcode_filter.set_name(_("G-code files"))
        gcode_filter.add_mime_type("text/x.gcode")
        filter_list.append(gcode_filter)

        # Set the filters for the dialog
        dialog.set_filters(filter_list)
        dialog.set_default_filter(gcode_filter)

        # Show the dialog and handle the response
        dialog.save(self, None, self.on_save_dialog_response)

    def on_home_clicked(self, action, param):
        if not config.machine:
            return
        driver = config.machine.driver
        task_mgr.add_coroutine(lambda ctx: driver.home())

    def on_frame_clicked(self, action, param):
        if not config.machine:
            return

        async def frame_coro(context: ExecutionContext):
            machine = config.machine
            if not machine:
                return

            try:
                head = machine.heads[0]
                if not head.frame_power:
                    return

                ops = await generate_job_ops(
                    self.doc, machine, self.surface.ops_generator, context
                )
                frame = ops.get_frame(
                    power=head.frame_power, speed=machine.max_travel_speed
                )
                frame *= 20
                await machine.driver.run(frame, machine)
            except Exception:
                logger.error("Failed to execute framing job", exc_info=True)
                raise

        task_mgr.add_coroutine(frame_coro, key="frame-job")

    def on_send_clicked(self, action, param):
        if not config.machine:
            return

        async def send_coro(context: ExecutionContext):
            machine = config.machine
            if not machine:
                return

            try:
                ops = await generate_job_ops(
                    self.doc, machine, self.surface.ops_generator, context
                )
                await machine.driver.run(ops, machine)
            except Exception:
                logger.error("Failed to send job to machine", exc_info=True)
                raise

        task_mgr.add_coroutine(send_coro, key="send-job")

    def on_hold_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        """
        Handles the 'change-state' signal for the 'hold' action.
        This is the correct handler for a stateful action.
        """
        if not config.machine:
            return
        driver = config.machine.driver
        is_requesting_hold = value.get_boolean()
        task_mgr.add_coroutine(lambda ctx: driver.set_hold(is_requesting_hold))
        action.set_state(value)

    def on_cancel_clicked(self, action, param):
        if not config.machine:
            return
        driver = config.machine.driver
        task_mgr.add_coroutine(lambda ctx: driver.cancel())

    def on_align_h_center_clicked(self, action, param):
        self.surface.center_horizontally()

    def on_align_v_center_clicked(self, action, param):
        self.surface.center_vertically()

    def on_align_left_clicked(self, action, param):
        self.surface.align_left()

    def on_align_right_clicked(self, action, param):
        self.surface.align_right()

    def on_align_top_clicked(self, action, param):
        self.surface.align_top()

    def on_align_bottom_clicked(self, action, param):
        self.surface.align_bottom()

    def on_spread_horizontally_clicked(self, action, param):
        self.surface.spread_horizontally()

    def on_spread_vertically_clicked(self, action, param):
        self.surface.spread_vertically()

    def on_save_dialog_response(self, dialog, result):
        try:
            file = dialog.save_finish(result)
            if not file:
                return
            file_path = Path(file.get_path())
        except GLib.Error as e:
            logger.error(f"Error saving file: {e.message}")
            return

        def write_gcode_sync(path, gcode):
            """Blocking I/O function to be run in a thread."""
            with open(path, "w") as f:
                f.write(gcode)

        async def export_coro(context: ExecutionContext):
            machine = config.machine
            if not machine:
                return

            try:
                # 1. Generate Ops (async, reports progress)
                ops = await generate_job_ops(
                    self.doc, machine, self.surface.ops_generator, context
                )

                # 2. Encode G-code (sync, but usually fast)
                context.set_message("Encoding G-code...")
                encoder = GcodeEncoder.for_machine(machine)
                gcode = encoder.encode(ops, machine)

                # 3. Write to file (sync, potentially slow, run in thread)
                context.set_message(f"Saving to {file_path}...")
                await asyncio.to_thread(write_gcode_sync, file_path, gcode)

                context.set_message("Export complete!")
                context.set_progress(1.0)
                context.flush()

            except Exception:
                logger.error("Failed to export G-code", exc_info=True)
                raise  # Re-raise to be caught by the task manager

        # Add the coroutine to the task manager
        task_mgr.add_coroutine(export_coro, key="export-gcode")

    def on_file_dialog_response(self, dialog, result):
        try:
            # Get the selected file
            file = dialog.open_finish(result)
            if file:
                # Load the SVG file and convert it to a grayscale surface
                file_path = Path(file.get_path())
                file_info = file.query_info(
                    Gio.FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE,
                    Gio.FileQueryInfoFlags.NONE,
                    None,
                )
                mime_type = file_info.get_content_type()
                self.load_file(file_path, mime_type)
        except GLib.Error as e:
            logger.error(f"Error opening file: {e.message}")

    def load_file(self, filename: Path, mime_type: Optional[str]):
        try:
            renderer = renderer_by_mime_type[mime_type]
        except KeyError:
            # On Windows, the file dialog returns not the mime type,
            # but the file extension instead.
            try:
                ext = mime_type.lower() if mime_type else None
                renderer = renderer_by_extension[ext]
            except KeyError:
                logger.error(
                    f"No renderer found for {mime_type}. "
                    f"MIME types: {renderer_by_mime_type.keys()} "
                    f"Extensions: {renderer_by_extension.keys()} "
                )
                return

        wp = WorkPiece.from_file(filename, renderer)

        # Calculate and set a default size and position for the new workpiece
        if wp.pos is None or wp.size is None:
            wswidth_mm, wsheight_mm = self.surface.get_size()
            wp_width_nat_mm, wp_height_nat_mm = wp.get_default_size(
                wswidth_mm, wsheight_mm
            )

            # Determine the size to use in mm, scaling down if necessary to fit
            width_mm = wp_width_nat_mm
            height_mm = wp_height_nat_mm
            if width_mm > wswidth_mm or height_mm > wsheight_mm:
                scale_w = wswidth_mm / width_mm if width_mm > 0 else 1
                scale_h = wsheight_mm / height_mm if height_mm > 0 else 1
                scale = min(scale_w, scale_h)
                width_mm *= scale
                height_mm *= scale

            # Set the workpiece's size and centered position in mm
            wp.set_size(width_mm, height_mm)
            x_mm = (wswidth_mm - width_mm) / 2
            y_mm = (wsheight_mm - height_mm) / 2
            wp.set_pos(x_mm, y_mm)

        cmd_name = _("Import {name}").format(name=filename.name)
        command = ListItemCommand(
            owner_obj=self.doc.active_layer,
            item=wp,
            undo_command="remove_workpiece",
            redo_command="add_workpiece",
            name=cmd_name,
        )
        self.doc.history_manager.execute(command)

        # No workpiece is active after loading a new document,
        # so ensure the properties widget is hidden.
        self.workpiece_revealer.set_reveal_child(False)

    def on_cut_requested(self, sender, workpieces: List[WorkPiece]):
        """Handles the 'cut-requested' signal from the WorkSurface."""
        if not workpieces:
            return

        self.on_copy_requested(sender, workpieces)
        # For a cut, the next paste should be at the original location
        # (no offset).
        self._paste_counter = 0

        history = self.doc.history_manager
        with history.transaction(_("Cut workpiece(s)")) as t:
            for wp in workpieces:
                cmd_name = _("Cut {name}").format(name=wp.name)
                command = ListItemCommand(
                    owner_obj=self.doc,
                    item=wp,
                    undo_command="add_workpiece",
                    redo_command="remove_workpiece",
                    name=cmd_name,
                )
                t.execute(command)

    def on_copy_requested(self, sender, workpieces: List[WorkPiece]):
        """
        Handles the 'copy-requested' signal. This snapshots the current
        state of the selected workpieces and resets the paste sequence.
        """
        if not workpieces:
            return
        # Create a snapshot of the current state by serializing to dicts.
        self._clipboard_snapshot = [wp.to_dict() for wp in workpieces]
        # For a copy, the next paste should be offset.
        self._paste_counter = 1
        logger.debug(
            f"Copied {len(self._clipboard_snapshot)} workpieces. "
            "Paste counter set to 1."
        )

    def on_paste_requested(self, sender, *args):
        """
        Handles the 'paste-requested' signal. Pastes a new set of items
        with a cumulative offset from the original clipboard snapshot.
        For a cut operation, the first paste is at the original location.
        """
        if not self._clipboard_snapshot:
            return

        history = self.doc.history_manager
        newly_pasted_workpieces = []

        with history.transaction(_("Paste workpiece(s)")) as t:
            # The paste counter determines the offset level.
            # It's 0 for the first paste of a cut, and >0 for all others.
            offset_x = self._paste_increment_mm[0] * self._paste_counter
            offset_y = self._paste_increment_mm[1] * self._paste_counter

            for wp_dict in self._clipboard_snapshot:
                new_wp = WorkPiece.from_dict(wp_dict)
                new_wp.uid = str(uuid.uuid4())
                newly_pasted_workpieces.append(new_wp)

                original_pos = wp_dict.get("pos")
                if original_pos:
                    new_wp.set_pos(
                        original_pos[0] + offset_x, original_pos[1] + offset_y
                    )

                cmd_name = _("Paste {name}").format(name=new_wp.name)
                command = ListItemCommand(
                    owner_obj=self.doc.active_layer,
                    item=new_wp,
                    undo_command="remove_workpiece",
                    redo_command="add_workpiece",
                    name=cmd_name,
                )
                t.execute(command)

        # Increment the counter for the *next* paste operation.
        self._paste_counter += 1

        if newly_pasted_workpieces:
            self.surface.select_workpieces(newly_pasted_workpieces)

    def on_duplicate_requested(self, sender, workpieces: List[WorkPiece]):
        """
        Handles the 'duplicate-requested' signal. This creates an exact
        copy of the selected workpieces in the same location.
        """
        if not workpieces:
            return

        history = self.doc.history_manager
        newly_duplicated_workpieces = []

        with history.transaction(_("Duplicate workpiece(s)")) as t:
            for wp in workpieces:
                wp_dict = wp.to_dict()
                new_wp = WorkPiece.from_dict(wp_dict)
                new_wp.uid = str(uuid.uuid4())
                newly_duplicated_workpieces.append(new_wp)

                cmd_name = _("Duplicate {name}").format(name=new_wp.name)
                command = ListItemCommand(
                    owner_obj=self.doc,
                    item=new_wp,
                    undo_command="remove_workpiece",
                    redo_command="add_workpiece",
                    name=cmd_name,
                )
                t.execute(command)

        if newly_duplicated_workpieces:
            self.surface.select_workpieces(newly_duplicated_workpieces)

    def on_menu_cut(self, action, param):
        selection = self.surface.get_selected_workpieces()
        if selection:
            self.on_cut_requested(self.surface, selection)

    def on_menu_copy(self, action, param):
        selection = self.surface.get_selected_workpieces()
        if selection:
            self.on_copy_requested(self.surface, selection)

    def on_menu_duplicate(self, action, param):
        selection = self.surface.get_selected_workpieces()
        if selection:
            self.on_duplicate_requested(self.surface, selection)

    def on_menu_remove(self, action, param):
        workpieces = self.surface.get_selected_workpieces()
        if not workpieces:
            return
        history = self.doc.history_manager
        with history.transaction(_("Remove workpiece(s)")) as t:
            for wp in workpieces:
                cmd_name = _("Remove {name}").format(name=wp.name)
                command = ListItemCommand(
                    owner_obj=self.doc,
                    item=wp,
                    undo_command="add_workpiece",
                    redo_command="remove_workpiece",
                    name=cmd_name,
                )
                t.execute(command)

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
