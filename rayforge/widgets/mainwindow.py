import asyncio
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from gi.repository import Gtk, Gio, GLib, Gdk, Adw, GObject  # type: ignore
from .. import __version__
from ..tasker.context import ExecutionContext
from ..config import config, task_mgr, machine_mgr
from ..driver import get_driver_cls
from ..driver.driver import driver_mgr, DeviceStatus
from ..driver.dummy import NoDeviceDriver
from ..util.resources import get_icon
from ..models.machine import Machine
from ..models.doc import Doc
from ..models.workpiece import WorkPiece
from ..opsencoder.gcode import GcodeEncoder
from ..render import renderers, renderer_by_mime_type, renderer_by_extension
from ..undo.list_cmd import ListItemCommand, ReorderListCommand
from .workflowview import WorkflowView
from .workbench.surface import WorkSurface
from .layerlist import LayerListView
from .statusview import (
    ConnectionStatusMonitor,
    TransportStatus,
    MachineStatusMonitor,
)
from .machineview import MachineView
from .preferencesdialog import PreferencesWindow
from .machinesettings import MachineSettingsDialog
from .progress import TaskProgressBar
from .workpieceprops import WorkpiecePropertiesWidget
from .canvas import CanvasElement
from .undobutton import UndoButton, RedoButton
from .about import AboutDialog


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


# This allows the plain Python Machine object to be stored in a Gio.ListStore.
class MachineListItem(GObject.Object):
    __gtype_name__ = "MachineListItem"

    def __init__(self, machine: Machine):
        super().__init__()
        self.machine = machine


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_title(_("Rayforge"))

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
                int(geometry.width * 0.6), int(geometry.height * 0.6)
            )
        else:
            self.set_default_size(1100, 800)

        # Setup keyboard actions.
        self._setup_actions()
        self._set_accelerators()

        # HeaderBar with left-aligned menu and centered title
        header_bar = Adw.HeaderBar()
        vbox.append(header_bar)

        # Create the menu model and the popover menubar
        menu_model = self._create_menu_model()
        menubar = Gtk.PopoverMenuBar.new_from_model(menu_model)
        menubar.add_css_class("in-header-menubar")
        header_bar.pack_start(menubar)  # Pack menubar to the left

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

        # Create a toolbar
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toolbar.set_margin_bottom(2)
        toolbar.set_margin_top(2)
        toolbar.set_margin_start(12)
        toolbar.set_margin_end(12)
        vbox.append(toolbar)

        # Import and export icons
        open_button = Gtk.Button()
        open_button.set_child(get_icon("document-open-symbolic"))
        open_button.set_tooltip_text(_("Import image"))
        open_button.connect("clicked", self.on_open_clicked)
        toolbar.append(open_button)

        self.export_button = Gtk.Button()
        self.export_button.set_child(get_icon("document-save-symbolic"))
        self.export_button.set_tooltip_text(_("Generate G-code"))
        self.export_button.connect("clicked", self.on_export_clicked)
        toolbar.append(self.export_button)

        # Undo/Redo Buttons
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        toolbar.append(sep)

        self.undo_button = UndoButton()
        toolbar.append(self.undo_button)

        self.redo_button = RedoButton()
        toolbar.append(self.redo_button)

        # Clear and visibility
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        toolbar.append(sep)

        clear_button = Gtk.Button()
        clear_button.set_child(get_icon("edit-clear-all-symbolic"))
        clear_button.set_tooltip_text(_("Remove all workpieces"))
        clear_button.connect("clicked", self.on_clear_clicked)
        toolbar.append(clear_button)

        self.visibility_on_icon = get_icon("stock-eye-symbolic")
        self.visibility_off_icon = get_icon("eye-not-looking-symbolic")
        button = Gtk.ToggleButton()
        button.set_active(True)
        button.set_child(self.visibility_on_icon)
        button.set_tooltip_text(_("Toggle workpiece visibility"))
        toolbar.append(button)
        button.connect("clicked", self.on_button_visibility_clicked)

        # Camera Image Visibility Toggle Button
        self.camera_visibility_on_icon = get_icon("camera-app-symbolic")
        self.camera_visibility_off_icon = get_icon("camera-disabled-symbolic")
        self.camera_visibility_button = Gtk.ToggleButton()
        self.camera_visibility_button.set_active(True)
        self.camera_visibility_button.set_child(self.camera_visibility_on_icon)
        self.camera_visibility_button.set_tooltip_text(
            _("Toggle camera image visibility")
        )
        self.camera_visibility_button.connect(
            "toggled", self._on_camera_image_visibility_toggled
        )
        toolbar.append(self.camera_visibility_button)

        # Show Travel Moves Toggle Button
        self.show_travel_button = Gtk.ToggleButton()
        self.show_travel_button.set_child(get_icon("function-linear-symbolic"))
        self.show_travel_button.set_active(False)
        self.show_travel_button.set_tooltip_text(
            _("Toggle travel move visibility")
        )
        self.show_travel_button.connect(
            "toggled", self._on_show_travel_toggled
        )
        toolbar.append(self.show_travel_button)

        # Control buttons: home, send, pause, stop
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        toolbar.append(sep)

        self.home_button = Gtk.Button()
        self.home_button.set_child(get_icon("go-home-symbolic"))
        self.home_button.set_tooltip_text(_("Home the machine"))
        self.home_button.connect("clicked", self.on_home_clicked)
        toolbar.append(self.home_button)

        self.frame_button = Gtk.Button()
        self.frame_button.set_child(get_icon("edit-select-all-symbolic"))
        self.frame_button.set_tooltip_text(
            _("Cycle laser head around the occupied area")
        )
        self.frame_button.connect("clicked", self.on_frame_clicked)
        toolbar.append(self.frame_button)

        self.send_button = Gtk.Button()
        self.send_button.set_child(get_icon("document-send-symbolic"))
        self.send_button.set_tooltip_text(_("Send to machine"))
        self.send_button.connect("clicked", self.on_send_clicked)
        toolbar.append(self.send_button)

        self.hold_on_icon = get_icon("pause-symbolic")
        self.hold_off_icon = get_icon("pause-symbolic")
        self.hold_button = Gtk.ToggleButton()
        self.hold_button.set_child(self.hold_off_icon)
        self.hold_button.set_tooltip_text(_("Pause machine"))
        self.hold_button.connect("clicked", self.on_hold_clicked)
        toolbar.append(self.hold_button)

        self.cancel_button = Gtk.Button()
        self.cancel_button.set_child(get_icon("process-stop-symbolic"))
        self.cancel_button.set_tooltip_text(_("Cancel running job"))
        self.cancel_button.connect("clicked", self.on_cancel_clicked)
        toolbar.append(self.cancel_button)

        # Add spacer to push machine selector to the right
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        toolbar.append(spacer)

        # Add clickable warning for misconfigured machine
        self.machine_warning_box = Gtk.Box(spacing=6)
        self.machine_warning_box.set_margin_end(12)
        warning_icon = Gtk.Image.new_from_icon_name("dialog-warning-symbolic")
        warning_label = Gtk.Label(label=_("Machine not fully configured"))
        warning_label.add_css_class("warning-label")
        self.machine_warning_box.append(warning_icon)
        self.machine_warning_box.append(warning_label)
        self.machine_warning_box.set_tooltip_text(
            _("Machine driver is missing required settings. Click to edit.")
        )
        self.machine_warning_box.set_visible(False)
        warning_click = Gtk.GestureClick.new()
        warning_click.connect("pressed", self._on_machine_warning_clicked)
        self.machine_warning_box.add_controller(warning_click)
        toolbar.append(self.machine_warning_box)

        # Add machine selector dropdown
        self._setup_machine_selector()
        toolbar.append(self.machine_dropdown)

        # Create the Paned splitting the window into left and right sections.
        self.paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self.paned.set_vexpand(True)
        vbox.append(self.paned)

        # Apply styles
        self.paned.add_css_class("mainpaned")
        provider = Gtk.CssProvider()
        provider.load_from_string(css)
        display = Gdk.Display.get_default()
        Gtk.StyleContext.add_provider_for_display(
            display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # Create a work area to display the image and paths
        width_mm, height_mm = config.machine.dimensions
        ratio = width_mm / height_mm
        self.frame = Gtk.AspectFrame(ratio=ratio, obey_child=False)
        self.frame.set_margin_start(12)
        self.frame.set_hexpand(True)
        self.paned.set_start_child(self.frame)

        # Make a default document.
        self.doc = Doc()
        self.doc.changed.connect(self.on_doc_changed)
        self.doc.active_layer_changed.connect(self._on_active_layer_changed)
        self.doc.history_manager.changed.connect(self.on_history_changed)

        self.surface = WorkSurface(
            self.doc,
            config.machine,
            cam_visible=self.camera_visibility_button.get_active(),
        )
        self.surface.set_hexpand(True)
        self.frame.set_child(self.surface)

        # Connect the undo/redo buttons to the document's history manager
        self.undo_button.set_history_manager(self.doc.history_manager)
        self.redo_button.set_history_manager(self.doc.history_manager)

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
        self.workflowview = WorkflowView(initial_workflow)
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

        # Create a two-row progress and status widget.
        self.progress_widget = TaskProgressBar(task_mgr)
        self.progress_widget.add_css_class("statusbar")
        vbox.append(self.progress_widget)

        # Get the top row of the widget to add status monitors to it.
        status_row = self.progress_widget.status_box
        status_row.set_margin_start(12)
        status_row.set_margin_end(12)

        # Monitor machine status
        label = Gtk.Label()
        label.set_markup(_("<b>Machine status:</b>"))
        status_row.append(label)

        self.machine_status = MachineStatusMonitor()
        status_row.append(self.machine_status)
        self.machine_status.changed.connect(self.on_machine_status_changed)

        # Monitor connection status
        label = Gtk.Label()
        label.set_markup(_("<b>Connection status:</b>"))
        label.set_margin_start(12)
        status_row.append(label)

        self.connection_status = ConnectionStatusMonitor()
        status_row.append(self.connection_status)
        self.connection_status.changed.connect(
            self.on_connection_status_changed
        )

        # Open machine log if the status row is clicked.
        gesture = Gtk.GestureClick()
        gesture.connect("pressed", self.on_status_bar_clicked, status_row)
        status_row.add_controller(gesture)

        # Set up driver and config signals.
        self._try_driver_setup()
        config.changed.connect(self.on_config_changed)
        driver_mgr.changed.connect(self.on_driver_changed)
        task_mgr.tasks_updated.connect(self.on_running_tasks_changed)
        self.needs_homing = config.machine.home_on_start

        # Set initial state
        self.on_config_changed(None)

    def _on_root_click_pressed(self, gesture, n_press, x, y):
        """
        Global click handler to unfocus widgets when clicking on "dead space".
        """
        self.surface.grab_focus()

    def _setup_machine_selector(self):
        """Creates the Gtk.DropDown for machine selection."""
        # The model holds GObject wrappers for our Machine objects.
        self.machine_model = Gio.ListStore.new(MachineListItem)

        # The Factory for the list items in the popup
        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_machine_selector_setup)
        factory.connect("bind", self._on_machine_selector_bind)

        # The Expression for the main dropdown button
        expression = Gtk.ClosureExpression.new(
            str, lambda item: item.machine.name if item else "", None
        )

        # Create the DropDown and set the factory
        self.machine_dropdown = Gtk.DropDown.new(
            self.machine_model, expression
        )
        self.machine_dropdown.set_factory(factory)

        self.machine_dropdown.set_tooltip_text(_("Select active machine"))
        self.machine_dropdown.connect(
            "notify::selected-item", self._on_machine_selected
        )

        # Connect to the machine manager to keep the list updated.
        machine_mgr.machine_added.connect(self._update_machine_list)
        machine_mgr.machine_removed.connect(self._update_machine_list)
        machine_mgr.machine_updated.connect(self._update_machine_list)

        # Initial population of the list.
        self._update_machine_list()

    def _on_machine_selector_setup(self, factory, list_item):
        """Setup a list item for the machine dropdown."""
        box = Gtk.Box(spacing=6)
        label = Gtk.Label()
        box.append(label)
        list_item.set_child(box)

    def _on_machine_selector_bind(self, factory, list_item):
        """Bind a machine object to a list item."""
        box = list_item.get_child()
        label = box.get_first_child()
        list_item_obj = list_item.get_item()
        label.set_text(list_item_obj.machine.name)

    def _update_machine_list(self, *args, **kwargs):
        """
        Repopulates the machine dropdown from the machine manager.
        """
        logger.debug("Updating machine list in dropdown.")
        machines = sorted(machine_mgr.machines.values(), key=lambda m: m.name)

        # Block the selection signal while we modify the list to avoid
        # recursion.
        self.machine_dropdown.handler_block_by_func(self._on_machine_selected)

        try:
            self.machine_model.remove_all()
            selected_index = -1
            for i, machine in enumerate(machines):
                # Append an instance of the wrapper, not the raw object
                self.machine_model.append(MachineListItem(machine))
                if config.machine and machine.id == config.machine.id:
                    selected_index = i

            if selected_index != -1:
                self.machine_dropdown.set_selected(selected_index)
        finally:
            # Unblock the signal in a finally block to ensure it always runs.
            self.machine_dropdown.handler_unblock_by_func(
                self._on_machine_selected
            )

        # Set visibility based on the number of machines
        self.machine_dropdown.set_visible(len(machines) > 1)

    def _on_machine_selected(self, dropdown, param):
        """Handles when a user selects a new machine from the dropdown."""
        selected_list_item = dropdown.get_selected_item()
        if selected_list_item:
            # selected_list_item is the MachineListItem wrapper
            logger.info(
                f"User selected machine: {selected_list_item.machine.name}"
            )
            config.set_machine(selected_list_item.machine)

    def _setup_actions(self):
        """Creates all Gio.SimpleActions for the window and application."""
        # File actions
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", self.on_quit_action)
        self.add_action(quit_action)

        import_action = Gio.SimpleAction.new("import", None)
        import_action.connect("activate", self.on_open_clicked)
        self.add_action(import_action)

        self.export_action = Gio.SimpleAction.new("export", None)
        self.export_action.connect("activate", self.on_export_clicked)
        self.add_action(self.export_action)

        # Edit actions
        self.undo_action = Gio.SimpleAction.new("undo", None)
        self.undo_action.connect(
            "activate", lambda a, p: self.doc.history_manager.undo()
        )
        self.add_action(self.undo_action)

        self.redo_action = Gio.SimpleAction.new("redo", None)
        self.redo_action.connect(
            "activate", lambda a, p: self.doc.history_manager.redo()
        )
        self.add_action(self.redo_action)

        self.cut_action = Gio.SimpleAction.new("cut", None)
        self.cut_action.connect("activate", self.on_menu_cut)
        self.add_action(self.cut_action)

        self.copy_action = Gio.SimpleAction.new("copy", None)
        self.copy_action.connect("activate", self.on_menu_copy)
        self.add_action(self.copy_action)

        self.paste_action = Gio.SimpleAction.new("paste", None)
        self.paste_action.connect("activate", self.on_paste_requested)
        self.add_action(self.paste_action)

        self.duplicate_action = Gio.SimpleAction.new("duplicate", None)
        self.duplicate_action.connect("activate", self.on_menu_duplicate)
        self.add_action(self.duplicate_action)

        self.remove_action = Gio.SimpleAction.new("remove", None)
        self.remove_action.connect("activate", self.on_menu_remove)
        self.add_action(self.remove_action)

        preferences_action = Gio.SimpleAction.new("preferences", None)
        preferences_action.connect("activate", self.show_preferences)
        self.add_action(preferences_action)

        # Help action
        about_action = Gio.SimpleAction.new("about", None)
        about_action.connect("activate", self.show_about_dialog)
        self.add_action(about_action)

    def _create_menu_model(self) -> Gio.Menu:
        """Creates the Gio.Menu model for the menubar."""
        menu_model = Gio.Menu()

        # File Menu
        file_menu = Gio.Menu()
        file_menu.append(_("Import..."), "win.import")
        file_menu.append(_("Export G-code..."), "win.export")

        quit_command = Gio.Menu()
        quit_command.append(_("Quit"), "win.quit")
        file_menu.append_section(None, quit_command)
        menu_model.append_submenu(_("_File"), file_menu)

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

        menu_model.append_submenu(_("_Edit"), edit_menu)

        # Help Menu
        help_menu = Gio.Menu()
        help_menu.append(_("About"), "win.about")
        menu_model.append_submenu(_("_Help"), help_menu)

        return menu_model

    def _set_accelerators(self):
        """Sets keyboard accelerators for the application's actions."""
        app = self.get_application()
        if not app:
            logger.warning(
                "Cannot set accelerators without a Gtk.Application."
            )
            return

        app.set_accels_for_action("win.import", ["<Primary>o"])
        app.set_accels_for_action("win.export", ["<Primary>e"])
        app.set_accels_for_action("win.quit", ["<Primary>q"])
        app.set_accels_for_action("win.undo", ["<Primary>z"])
        app.set_accels_for_action(
            "win.redo", ["<Primary>y", "<Primary><Shift>z"]
        )
        app.set_accels_for_action("win.cut", ["<Primary>x"])
        app.set_accels_for_action("win.copy", ["<Primary>c"])
        app.set_accels_for_action("win.paste", ["<Primary>v"])
        app.set_accels_for_action("win.duplicate", ["<Primary>d"])
        app.set_accels_for_action("win.remove", ["Delete"])
        app.set_accels_for_action("win.preferences", ["<Primary>comma"])
        app.set_accels_for_action("win.about", ["F1"])

    def _try_driver_setup(self):
        # Reconfigure, because params may have changed.
        driver_name = config.machine.driver
        if driver_name is None:
            logger.warning("No driver configured.")
            return
        driver_cls = get_driver_cls(driver_name)
        try:
            # This wrapper coroutine adapts the call to the TaskManager's
            # expectation that all tasks accept an ExecutionContext.
            async def setup_driver_coro(
                context: ExecutionContext, cls, **kwargs
            ):
                # The context is accepted but not used for this simple task.
                await driver_mgr.select_by_cls(cls, **kwargs)

            task_mgr.add_coroutine(
                setup_driver_coro,
                driver_cls,
                key="driver-setup",
                **config.machine.driver_args,
            )
        except Exception as e:
            logger.error(f"Failed to set up driver: {e}")
            return

    def on_driver_changed(self, sender, driver):
        self.update_state()

    def on_machine_status_changed(self, sender):
        # If the machine is idle for the first time, perform auto-homing
        # if requested.
        if self.needs_homing and driver_mgr.driver:
            device_status = self.machine_status.get_status()
            if device_status == DeviceStatus.IDLE:
                self.needs_homing = False
                task_mgr.add_coroutine(driver_mgr.driver.home)

        self.update_state()

    def on_connection_status_changed(self, sender):
        self.update_state()

    def on_history_changed(self, history_manager):
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

    def on_config_changed(self, sender, **kwargs):
        self.surface.set_machine(config.machine)
        width_mm, height_mm = config.machine.dimensions
        self.frame.set_ratio(width_mm / height_mm)

        # Apply selected device driver.
        self._try_driver_setup()
        self.surface.update_from_doc(self.doc)
        self.update_state()

        # Update theme
        self.apply_theme()

        # Ensure dropdown selection is synced with config
        self._update_machine_list()

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
        device_status = self.machine_status.get_status()
        has_tasks = len(task_mgr._tasks) > 0
        can_export = self.doc.has_workpiece() and not has_tasks
        has_selection = len(self.surface.get_selected_workpieces()) > 0
        can_undo = self.doc.history_manager.can_undo()
        can_redo = self.doc.history_manager.can_redo()
        can_paste = len(self._clipboard_snapshot) > 0

        # Show warning if the current machine is not configured
        warning_visible = False
        if driver_mgr.driver and not driver_mgr.driver.did_setup:
            warning_visible = True
        self.machine_warning_box.set_visible(warning_visible)

        # Update action sensitivity
        self.export_action.set_enabled(can_export)
        self.undo_action.set_enabled(can_undo)
        self.redo_action.set_enabled(can_redo)
        self.cut_action.set_enabled(has_selection and not has_tasks)
        self.copy_action.set_enabled(has_selection)
        self.paste_action.set_enabled(can_paste and not has_tasks)
        self.duplicate_action.set_enabled(has_selection and not has_tasks)
        self.remove_action.set_enabled(has_selection and not has_tasks)

        # Update button sensitivity
        self.export_button.set_sensitive(can_export)
        self.export_button.set_tooltip_text(
            _("Cannot export while operations are being generated")
            if has_tasks
            else _("Generate G-code")
        )

        self.home_button.set_sensitive(device_status == DeviceStatus.IDLE)

        can_frame = (
            config.machine.can_frame()
            and self.doc.has_result()
            and device_status == DeviceStatus.IDLE
            and not has_tasks
        )
        self.frame_button.set_sensitive(can_frame)
        self.frame_button.set_tooltip_text(
            _("Cannot frame while operations are being generated")
            if has_tasks
            else _("Cycle laser head around the occupied area")
        )

        conn_status = self.connection_status.get_status()
        send_sensitive = True
        send_tooltip = _("Send to machine")
        if driver_mgr.driver.__class__ is NoDeviceDriver:
            send_tooltip = _("Send to machine (select driver to enable)")
            send_sensitive = False
        elif not (driver_mgr.driver and driver_mgr.driver.did_setup):
            send_tooltip = _("Send to machine (configure driver to enable)")
            send_sensitive = False
        elif conn_status != TransportStatus.CONNECTED:
            send_tooltip = _("Send to machine (connect to enable)")
            send_sensitive = False
        elif has_tasks:
            send_tooltip = _(
                "Send to machine (wait for calculations to finish)"
            )
            send_sensitive = False
        elif not self.doc.has_result():
            send_sensitive = False
        self.send_button.set_sensitive(send_sensitive)
        self.send_button.set_tooltip_text(send_tooltip)

        hold_sensitive = device_status in (DeviceStatus.RUN, DeviceStatus.HOLD)
        self.hold_button.set_sensitive(hold_sensitive)
        self.hold_button.set_active(device_status == DeviceStatus.HOLD)

        cancel_sensitive = device_status in (
            DeviceStatus.RUN,
            DeviceStatus.HOLD,
            DeviceStatus.JOG,
            DeviceStatus.CYCLE,
        )
        self.cancel_button.set_sensitive(cancel_sensitive)

        # Laser dot
        connected = conn_status == TransportStatus.CONNECTED
        self.surface.set_laser_dot_visible(connected)
        state = self.machine_status.state
        if not state:
            return
        x, y = state.machine_pos[:2]
        if x is not None and y is not None:
            self.surface.set_laser_dot_position(x, y)

    def _on_machine_warning_clicked(self, *args):
        """Opens the machine settings dialog for the current machine."""
        current_machine = config.machine
        if not current_machine:
            return

        dialog = MachineSettingsDialog(machine=current_machine)
        dialog.present(self)

    def on_status_bar_clicked(self, gesture, n_press, x, y, box):
        dialog = MachineView()
        dialog.notification_requested.connect(self._on_dialog_notification)
        dialog.present(self)

    def _on_dialog_notification(self, sender, message: str = ""):
        """Shows a toast when requested by a child dialog."""
        self.toast_overlay.add_toast(Adw.Toast.new(message))

    def on_quit_action(self, action, parameter):
        self.close()

    def on_open_clicked(self, action, param=None):
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

    def on_button_visibility_clicked(self, button):
        is_active = button.get_active()
        self.surface.set_workpieces_visible(is_active)
        if is_active:
            button.set_child(self.visibility_on_icon)
        else:
            button.set_child(self.visibility_off_icon)

    def _on_camera_image_visibility_toggled(self, button):
        is_active = button.get_active()
        self.surface.set_camera_image_visibility(is_active)
        if is_active:
            button.set_child(self.camera_visibility_on_icon)
        else:
            button.set_child(self.camera_visibility_off_icon)

    def _on_show_travel_toggled(self, button):
        is_active = button.get_active()
        self.surface.set_show_travel_moves(is_active)

    def on_clear_clicked(self, button):
        if not self.doc.workpieces:
            return

        command = ReorderListCommand(
            target_obj=self.doc,
            list_property_name="workpieces",
            new_list=[],
            setter_method_name="set_workpieces",
            name=_("Remove all workpieces"),
        )
        self.doc.history_manager.execute(command)

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

    def on_home_clicked(self, button):
        if not driver_mgr.driver:
            return
        task_mgr.add_coroutine(driver_mgr.driver.home)

    def on_frame_clicked(self, button):
        if not driver_mgr.driver:
            return

        async def frame_coro(context: ExecutionContext):
            try:
                head = config.machine.heads[0]
                if not head.frame_power:
                    return

                ops = await self.doc.generate_job_ops(context)
                frame = ops.get_frame(
                    power=head.frame_power,
                    speed=config.machine.max_travel_speed,
                )
                frame *= 20  # cycle 20 times
                if not driver_mgr.driver:
                    raise RuntimeError("No driver configured for framing.")
                await driver_mgr.driver.run(frame, config.machine)
            except Exception:
                logger.error("Failed to execute framing job", exc_info=True)
                raise

        task_mgr.add_coroutine(
            frame_coro,
            key="frame-job",
        )

    def on_send_clicked(self, button):
        if not driver_mgr.driver:
            return

        async def send_coro(context: ExecutionContext):
            try:
                ops = await self.doc.generate_job_ops(context)
                if not driver_mgr.driver:
                    raise RuntimeError("No driver configured to send job.")
                await driver_mgr.driver.run(ops, config.machine)
            except Exception:
                logger.error("Failed to send job to machine", exc_info=True)
                raise

        task_mgr.add_coroutine(
            send_coro,
            key="send-job",
        )

    def on_hold_clicked(self, button):
        if not driver_mgr.driver:
            return
        if button.get_active():
            task_mgr.add_coroutine(driver_mgr.driver.set_hold)
            button.set_child(self.hold_on_icon)
        else:
            task_mgr.add_coroutine(driver_mgr.driver.set_hold, False)
            button.set_child(self.hold_off_icon)

    def on_cancel_clicked(self, button):
        if not driver_mgr.driver:
            return
        task_mgr.add_coroutine(driver_mgr.driver.cancel)

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
            try:
                # 1. Generate Ops (async, reports progress)
                ops = await self.doc.generate_job_ops(context)

                # 2. Encode G-code (sync, but usually fast)
                context.set_message("Encoding G-code...")
                encoder = GcodeEncoder.for_machine(config.machine)
                gcode = encoder.encode(ops, config.machine)

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
        task_mgr.add_coroutine(
            export_coro,
            key="export-gcode",
        )

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

    def _on_preferences_dialog_closed(self, dialog):
        logger.debug("Preferences dialog closed")
        self.surface.grab_focus()  # re-enables keyboard shortcuts
