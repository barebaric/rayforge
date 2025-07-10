from gi.repository import Gtk, Gio, GLib, Gdk, Adw  # type: ignore
from .. import __version__
from ..task import task_mgr
from ..config import config
from ..driver import get_driver_cls
from ..driver.driver import driver_mgr, DeviceStatus
from ..driver.dummy import NoDeviceDriver
from ..util.resources import get_icon
from ..models.doc import Doc
from ..models.workpiece import WorkPiece
from ..opsencoder.gcode import GcodeEncoder
from ..render import renderers, renderer_by_mime_type
from .workplanview import WorkPlanView
from .workbench.surface import WorkSurface
from .statusview import ConnectionStatusMonitor, \
                        TransportStatus, \
                        MachineStatusMonitor
from .machineview import MachineView
from .machinesettings import MachineSettingsDialog
from .progress import TaskProgressBar
from .workpieceprops import WorkpiecePropertiesWidget
from .canvas import CanvasElement
from .workbench.workpieceelem import WorkPieceElement


css = """
.mainpaned > separator {
    border: none;
    box-shadow: none;
}

.statusbar {
    border-radius: 5px;
    padding: 12px;
}

.statusbar:hover {
    background-color: @theme_hover_bg_color;
}

/* Style for the progress bar */
.statusbar > progressbar {
    background-color: alpha(@theme_fg_color, 0.1);
    border-color: alpha(@theme_fg_color, 0.3);
}

/* Style for the progress bar's trough */
.statusbar > progressbar > trough {
    background-color: alpha(@theme_fg_color, 0.2);
}

/* Style for the progress bar's progress indicator */
.statusbar > progressbar > trough > progress {
    background-color: alpha(@theme_fg_color, 0.4);
}
"""


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_title("Rayforge")

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
            self.set_default_size(int(geometry.width * 0.6),
                                  int(geometry.height * 0.6))
        else:
            self.set_default_size(1200, 900)

        # Define a "window quit" action.
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", self.on_quit_action)
        self.add_action(quit_action)

        # Create the main vbox
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.set_content(vbox)

        # Show the application header bar with hamburger menu
        header_bar = Adw.HeaderBar()
        vbox.append(header_bar)

        # Create a menu
        menu_button = Gtk.MenuButton()
        menu_button.set_icon_name("open-menu-symbolic")
        menu_model = Gio.Menu()
        menu_model.append("About", "win.about")
        menu_model.append("Preferences", "win.settings")
        menu_button.set_menu_model(menu_model)
        header_bar.pack_end(menu_button)

        # Add the "about" action
        about_action = Gio.SimpleAction.new("about", None)
        about_action.connect("activate", self.show_about_dialog)
        self.add_action(about_action)

        # Add the "quit" action
        settings_action = Gio.SimpleAction.new("settings", None)
        settings_action.connect("activate", self.show_machine_settings)
        self.add_action(settings_action)

        # Create a toolbar
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toolbar.set_margin_bottom(2)
        toolbar.set_margin_top(2)
        toolbar.set_margin_start(12)
        toolbar.set_margin_end(12)
        vbox.append(toolbar)

        # Import and export icons
        open_button = Gtk.Button()
        open_button.set_child(get_icon('open'))
        open_button.set_tooltip_text("Import image")
        open_button.connect("clicked", self.on_open_clicked)
        toolbar.append(open_button)

        self.export_button = Gtk.Button()
        self.export_button.set_child(get_icon('publish'))
        self.export_button.set_tooltip_text("Generate GCode")
        self.export_button.connect("clicked", self.on_export_clicked)
        toolbar.append(self.export_button)

        # Clear and visibility
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        toolbar.append(sep)

        clear_button = Gtk.Button()
        clear_button.set_child(get_icon('clear-layers'))
        clear_button.set_tooltip_text("Remove all workpieces")
        clear_button.connect("clicked", self.on_clear_clicked)
        toolbar.append(clear_button)

        self.visibility_on_icon = get_icon('visibility-on')
        self.visibility_off_icon = get_icon('visibility-off')
        button = Gtk.ToggleButton()
        button.set_active(True)
        button.set_child(self.visibility_on_icon)
        button.set_tooltip_text("Toggle workpiece visibility")
        toolbar.append(button)
        button.connect('clicked', self.on_button_visibility_clicked)

        # Show Travel Moves Toggle Button
        self.show_travel_button = Gtk.ToggleButton()
        self.show_travel_button.set_child(get_icon('timeline'))
        self.show_travel_button.set_active(False)
        self.show_travel_button.set_tooltip_text(
            "Toggle travel move visibility"
        )
        self.show_travel_button.connect(
            "toggled", self._on_show_travel_toggled
        )
        toolbar.append(self.show_travel_button)

        # Control buttons: home, send, pause, stop
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        toolbar.append(sep)

        self.home_button = Gtk.Button()
        self.home_button.set_child(get_icon('home'))
        self.home_button.set_tooltip_text("Home the machine")
        self.home_button.connect("clicked", self.on_home_clicked)
        toolbar.append(self.home_button)

        self.frame_button = Gtk.Button()
        self.frame_button.set_child(get_icon('frame'))
        self.frame_button.set_tooltip_text(
            "Cycle laser head around the occupied area"
        )
        self.frame_button.connect("clicked", self.on_frame_clicked)
        toolbar.append(self.frame_button)

        self.send_button = Gtk.Button()
        self.send_button.set_child(get_icon('send'))
        self.send_button.set_tooltip_text("Send to machine")
        self.send_button.connect("clicked", self.on_send_clicked)
        toolbar.append(self.send_button)

        self.hold_on_icon = get_icon('play-arrow')
        self.hold_off_icon = get_icon('pause')
        self.hold_button = Gtk.ToggleButton()
        self.hold_button.set_child(self.hold_off_icon)
        self.hold_button.set_tooltip_text("Pause machine")
        self.hold_button.connect("clicked", self.on_hold_clicked)
        toolbar.append(self.hold_button)

        self.cancel_button = Gtk.Button()
        self.cancel_button.set_child(get_icon('stop'))
        self.cancel_button.set_tooltip_text("Cancel running job")
        self.cancel_button.connect("clicked", self.on_cancel_clicked)
        toolbar.append(self.cancel_button)

        # Create the Paned splitting the window into left and right sections.
        self.paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        vbox.append(self.paned)

        # Apply styles
        self.paned.add_css_class("mainpaned")
        provider = Gtk.CssProvider()
        provider.load_from_data(css.encode())
        display = Gdk.Display.get_default()
        Gtk.StyleContext.add_provider_for_display(
            display,
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # Create a work area to display the image and paths
        width_mm, height_mm = config.machine.dimensions
        ratio = width_mm/height_mm
        self.frame = Gtk.AspectFrame(ratio=ratio, obey_child=True)
        self.frame.set_margin_start(12)
        self.frame.set_margin_end(12)
        self.frame.set_hexpand(True)
        self.frame.set_vexpand(True)
        self.paned.set_start_child(self.frame)

        self.surface = WorkSurface()
        self.surface.set_size(width_mm, height_mm)
        self.surface.set_hexpand(True)
        self.frame.set_child(self.surface)

        # Make a default document.
        self.doc = Doc()
        self.doc.changed.connect(self.on_doc_changed)

        # Create a vertical paned for the right pane content
        right_pane_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        right_pane_box.set_margin_top(6)
        right_pane_box.set_margin_bottom(12)
        self.paned.set_end_child(right_pane_box)
        self.paned.set_resize_end_child(False)
        self.paned.set_shrink_end_child(False)

        # Show the work plan.
        self.workplanview = WorkPlanView(self.doc.workplan)
        self.workplanview.set_size_request(400, -1)
        self.workplanview.set_vexpand(True)
        self.workplanview.set_margin_start(4)
        right_pane_box.append(self.workplanview)

        # Add the WorkpiecePropertiesWidget
        self.workpiece_props_widget = WorkpiecePropertiesWidget(None)
        workpiece_props_container = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL
        )
        workpiece_props_container.set_vexpand(False)
        workpiece_props_container.set_valign(Gtk.Align.END)
        workpiece_props_container.append(self.workpiece_props_widget)

        self.workpiece_revealer = Gtk.Revealer()
        self.workpiece_revealer.set_child(workpiece_props_container)
        self.workpiece_revealer.set_reveal_child(False)
        self.workpiece_revealer.set_transition_type(
            Gtk.RevealerTransitionType.SLIDE_UP
        )
        right_pane_box.append(self.workpiece_revealer)
        self.workpiece_props_widget.set_margin_top(20)
        self.workpiece_props_widget.set_margin_start(4)

        # Connect signals for workpiece selection
        self.surface.active_element_changed.connect(
            self._on_active_workpiece_changed
        )

        # Create a status bar.
        status_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        status_bar.set_hexpand(True)
        status_bar.set_halign(Gtk.Align.FILL)
        status_bar.set_margin_end(12)
        status_bar.get_style_context().add_class("statusbar")
        vbox.append(status_bar)

        # Add the TaskProgressBar to the status bar
        self.progress_bar = TaskProgressBar(task_mgr)
        status_bar.append(self.progress_bar)

        # Machine and connection status box.
        status_box_outer = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6
        )
        status_box_outer.set_size_request(500, -1)
        status_box_outer.set_hexpand(False)
        status_bar.append(status_box_outer)

        status_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6
        )
        status_box.set_halign(Gtk.Align.END)
        status_box.set_hexpand(True)
        status_box_outer.append(status_box)

        # Monitor machine status
        label = Gtk.Label()
        label.set_markup("<b>Machine status:</b>")
        status_box.append(label)

        self.machine_status = MachineStatusMonitor()
        status_box.append(self.machine_status)
        self.machine_status.changed.connect(
            self.on_machine_status_changed
        )

        # Monitor connection status
        label = Gtk.Label()
        label.set_markup("<b>Connection status:</b>")
        label.set_margin_start(12)
        status_box.append(label)

        self.connection_status = ConnectionStatusMonitor()
        status_box.append(self.connection_status)
        self.connection_status.changed.connect(
            self.on_connection_status_changed
        )

        # Open machine log if status bar is clicked.
        gesture = Gtk.GestureClick()
        gesture.connect("pressed", self.on_status_bar_clicked, status_bar)
        status_bar.add_controller(gesture)

        # Set up driver and config signals.
        self._try_driver_setup()
        config.changed.connect(self.on_config_changed)
        driver_mgr.changed.connect(self.on_driver_changed)
        self.needs_homing = config.machine.home_on_start

    def _try_driver_setup(self):
        # Reconfigure, because params may have changed.
        driver_cls = get_driver_cls(config.machine.driver)
        try:
            task_mgr.add_coroutine(driver_mgr.select_by_cls(
                driver_cls,
                **config.machine.driver_args
            ))
        except Exception as e:
            print("Failed to set up driver:", e)
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
                task_mgr.add_coroutine(driver_mgr.driver.home())

        self.update_state()

    def on_connection_status_changed(self, sender):
        self.update_state()

    def on_doc_changed(self, sender, **kwargs):
        self.surface.update_from_doc(self.doc)
        self.update_state()

    def _on_active_workpiece_changed(self, sender, element: CanvasElement):
        workpiece = (
            element.data if isinstance(element, WorkPieceElement) else None
        )
        self.workpiece_props_widget.set_workpiece(workpiece)
        self.workpiece_revealer.set_reveal_child(workpiece is not None)

    def on_config_changed(self, sender, **kwargs):
        self.surface.set_size(*config.machine.dimensions)
        width_mm, height_mm = config.machine.dimensions
        ratio = width_mm/height_mm
        self.frame.set_ratio(ratio)

        # Apply selected device driver.
        self._try_driver_setup()
        self.surface.update_from_doc(self.doc)
        self.update_state()

    def update_state(self):
        device_status = self.machine_status.get_status()

        # Update button states
        self.export_button.set_sensitive(self.doc.has_workpiece())
        self.home_button.set_sensitive(device_status == DeviceStatus.IDLE)

        # Frame button
        can_frame = config.machine.can_frame() and self.doc.has_result()
        can_frame = can_frame and device_status == DeviceStatus.IDLE
        self.frame_button.set_sensitive(can_frame)

        # Send button
        conn_status = self.connection_status.get_status()
        if driver_mgr.driver.__class__ is NoDeviceDriver:
            text = "Send to machine (select driver to enable)"
            sensitive = False
        elif conn_status != TransportStatus.CONNECTED:
            text = "Send to machine (connect to enable)"
            sensitive = False
        else:
            text = "Send to machine"
            sensitive = True
        self.send_button.set_sensitive(sensitive)
        self.send_button.set_tooltip_text(text)

        # Pause button
        sensitive = device_status in (DeviceStatus.RUN, DeviceStatus.HOLD)
        self.hold_button.set_sensitive(sensitive)
        self.hold_button.set_active(device_status == DeviceStatus.HOLD)

        # Cancel button
        sensitive = device_status in (
            DeviceStatus.RUN,
            DeviceStatus.HOLD,
            DeviceStatus.JOG,
            DeviceStatus.CYCLE,
        )
        self.cancel_button.set_sensitive(sensitive)

        # Laser dot
        connected = conn_status == TransportStatus.CONNECTED
        self.surface.set_laser_dot_visible(connected)
        state = self.machine_status.state
        if state and None not in state.machine_pos:
            self.surface.set_laser_dot_position(*state.machine_pos[:2])

    def on_status_bar_clicked(self, gesture, n_press, x, y, box):
        dialog = MachineView()
        dialog.present(self)

    def on_quit_action(self, action, parameter):
        self.close()

    def on_open_clicked(self, button):
        # Create a file chooser dialog
        dialog = Gtk.FileDialog.new()
        dialog.set_title("Open SVG File")

        # Create a Gio.ListModel for the filters
        filter_list = Gio.ListStore.new(Gtk.FileFilter)
        all_supported = Gtk.FileFilter()
        all_supported.set_name("All supported")
        for renderer in renderers:
            file_filter = Gtk.FileFilter()
            file_filter.set_name(renderer.label)
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
        self.surface.set_workpieces_visible(button.get_active())
        if button.get_active():
            button.set_child(self.visibility_on_icon)
        else:
            button.set_child(self.visibility_off_icon)

    def _on_show_travel_toggled(self, button):
        is_active = button.get_active()
        self.surface.set_show_travel_moves(is_active)

    def on_clear_clicked(self, button):
        self.surface.clear()

    def on_export_clicked(self, button):
        # Create a file chooser dialog for saving the file
        dialog = Gtk.FileDialog.new()
        dialog.set_title("Save G-code File")

        # Set the default file name
        dialog.set_initial_name("output.gcode")

        # Create a Gio.ListModel for the filters
        filter_list = Gio.ListStore.new(Gtk.FileFilter)
        gcode_filter = Gtk.FileFilter()
        gcode_filter.set_name("G-code files")
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
        task_mgr.add_coroutine(driver_mgr.driver.home())

    def on_frame_clicked(self, button):
        if not driver_mgr.driver:
            return
        try:
            head = config.machine.heads[0]
        except IndexError:
            return
        if not head.frame_power:
            return

        ops = self.doc.workplan.execute()
        frame = ops.get_frame(
            power=head.frame_power,
            speed=config.machine.max_travel_speed
        )
        frame *= 20  # cycle 20 times
        task_mgr.add_coroutine(driver_mgr.driver.run(frame, config.machine))

    def on_send_clicked(self, button):
        if not driver_mgr.driver:
            return
        ops = self.doc.workplan.execute()
        task_mgr.add_coroutine(driver_mgr.driver.run(ops, config.machine))

    def on_hold_clicked(self, button):
        if not driver_mgr.driver:
            return
        if button.get_active():
            task_mgr.add_coroutine(driver_mgr.driver.set_hold())
            button.set_child(self.hold_on_icon)
        else:
            task_mgr.add_coroutine(driver_mgr.driver.set_hold(False))
            button.set_child(self.hold_off_icon)

    def on_cancel_clicked(self, button):
        if not driver_mgr.driver:
            return
        task_mgr.add_coroutine(driver_mgr.driver.cancel())

    def on_save_dialog_response(self, dialog, result):
        try:
            file = dialog.save_finish(result)
            if not file:
                return
            file_path = file.get_path()

            # Serialize the G-code
            encoder = GcodeEncoder()
            ops = self.doc.workplan.execute()
            gcode = encoder.encode(ops, config.machine)

            # Write the G-code to the file
            with open(file_path, 'w') as f:
                f.write(gcode)
        except GLib.Error as e:
            print(f"Error saving file: {e.message}")

    def on_file_dialog_response(self, dialog, result):
        try:
            # Get the selected file
            file = dialog.open_finish(result)
            if file:
                # Load the SVG file and convert it to a grayscale surface
                file_path = file.get_path()
                file_info = file.query_info(
                    Gio.FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE,
                    Gio.FileQueryInfoFlags.NONE,
                    None
                )
                mime_type = file_info.get_content_type()
                self.load_file(file_path, mime_type)
        except GLib.Error as e:
            print(f"Error opening file: {e.message}")

    def load_file(self, filename, mime_type):
        renderer = renderer_by_mime_type[mime_type]
        wp = WorkPiece.from_file(filename, renderer)
        self.doc.add_workpiece(wp)
        self.surface.update_from_doc(self.doc)
        self.update_state()
        # No workpiece is active after loading a new document,
        # so ensure the properties widget is hidden.
        self.workpiece_revealer.set_reveal_child(False)

    def show_about_dialog(self, action, param):
        about_dialog = Adw.AboutDialog(
            application_name="Rayforge",
            application_icon="com.barebaric.rayforge",
            developer_name="Barebaric",
            version=__version__ or 'unknown',
            copyright="© 2025 Samuel Abels",
            website="https://github.com/barebaric/rayforge",
            issue_url="https://github.com/barebaric/rayforge/issues",
            developers=["Samuel Abels"],
            license_type=Gtk.License.MIT_X11
        )
        about_dialog.present(self)

    def show_machine_settings(self, action, param):
        dialog = MachineSettingsDialog(config.machine)
        dialog.present(self)
