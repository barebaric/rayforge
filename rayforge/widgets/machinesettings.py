from gi.repository import Gtk, Adw
from ..driver import drivers, get_driver_cls, get_params
from ..models.machine import Laser, Camera
from ..util.adwfix import get_spinrow_int, get_spinrow_float
from .dynamicprefs import DynamicPreferencesGroup
from .cameraproperties import CameraProperties
from .cameraselectiondialog import CameraSelectionDialog


class MachineSettingsDialog(Adw.PreferencesDialog):
    def __init__(self, machine, **kwargs):
        super().__init__(**kwargs)
        self.machine = machine

        # Make the dialog resizable
        self.set_size_request(-1, -1)

        # Create the "General" page (first page)
        general_page = Adw.PreferencesPage(
            title="General",
            icon_name='preferences-system-symbolic'
        )
        self.add(general_page)

        self.driver_group = DynamicPreferencesGroup(title="Driver Settings")
        self.driver_group.data_changed.connect(self.on_driver_param_changed)
        general_page.add(self.driver_group)

        # Driver selector
        self.driver_store = Gtk.StringList()
        for d in drivers:
            self.driver_store.append(d.label)
        driver_cls = get_driver_cls(machine.driver)
        self.combo_row = Adw.ComboRow(
            title=driver_cls.label if driver_cls else 'Select driver',
            subtitle=driver_cls.subtitle if driver_cls else None,
            model=self.driver_store
        )
        self.combo_row.set_use_subtitle(True)
        self.combo_row.set_subtitle(driver_cls.subtitle)
        self.driver_group.add(self.combo_row)
        self.driver_group.create_params(get_params(driver_cls))
        self.driver_group.set_values(machine.driver_args)

        # Set up a custom factory to display both title and subtitle
        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self.on_factory_setup)
        factory.connect("bind", self.on_factory_bind)
        self.combo_row.set_factory(factory)

        if driver_cls:
            selected_index = drivers.index(driver_cls)
            self.combo_row.set_selected(selected_index)

        # Connect to the "notify::selected" signal to handle selection changes
        self.combo_row.connect("notify::selected", self.on_combo_row_changed)

        # Group for Machine Settings
        machine_group = Adw.PreferencesGroup(title="Machine Settings")
        general_page.add(machine_group)

        home_on_start_row = Adw.SwitchRow()
        home_on_start_row.set_title("Home On Start")
        home_on_start_row.set_subtitle(
            "Whether Rayforce will send a homing command when it is started"
        )
        home_on_start_row.set_active(machine.home_on_start)
        home_on_start_row.connect(
            'notify::active',
            self.on_home_on_start_changed
        )
        machine_group.add(home_on_start_row)

        # Max Travel Speed
        travel_speed_adjustment = Gtk.Adjustment(
            value=self.machine.max_travel_speed,
            lower=0,
            upper=10000,
            step_increment=1,
            page_increment=10
        )
        self.travel_speed_row = Adw.SpinRow(
            title="Max Travel Speed",
            subtitle="Maximum travel speed in mm/min",
            adjustment=travel_speed_adjustment
        )
        self.travel_speed_row.connect("changed", self.on_travel_speed_changed)
        machine_group.add(self.travel_speed_row)

        # Max Cut Speed
        cut_speed_adjustment = Gtk.Adjustment(
            value=self.machine.max_cut_speed,
            lower=0,
            upper=10000,
            step_increment=1,
            page_increment=10
        )
        self.cut_speed_row = Adw.SpinRow(
            title="Max Cut Speed",
            subtitle="Maximum cutting speed in mm/min",
            adjustment=cut_speed_adjustment
        )
        self.cut_speed_row.connect("changed", self.on_cut_speed_changed)
        machine_group.add(self.cut_speed_row)

        # Dimensions
        dimensions_group = Adw.PreferencesGroup(title="Dimensions")
        general_page.add(dimensions_group)

        width_adjustment = Gtk.Adjustment(
            value=self.machine.dimensions[0],
            lower=50,
            upper=10000,
            step_increment=1,
            page_increment=10
        )
        self.width_row = Adw.SpinRow(
            title="Width",
            subtitle="Width of the machine in mm",
            adjustment=width_adjustment
        )
        self.width_row.connect("changed", self.on_width_changed)
        dimensions_group.add(self.width_row)

        height_adjustment = Gtk.Adjustment(
            value=self.machine.dimensions[1],
            lower=50,
            upper=10000,
            step_increment=1,
            page_increment=10
        )
        self.height_row = Adw.SpinRow(
            title="Height",
            subtitle="Height of the machine in mm",
            adjustment=height_adjustment
        )
        self.height_row.connect("changed", self.on_height_changed)
        dimensions_group.add(self.height_row)

        # Create the "GCode" page
        gcode_page = Adw.PreferencesPage(
            title="GCode",
            icon_name='applications-engineering-symbolic'
        )
        self.add(gcode_page)

        # Preamble
        preamble_group = Adw.PreferencesGroup(title="Preamble")
        gcode_page.add(preamble_group)
        self.preamble_entry = Gtk.TextView()
        self.preamble_entry.set_size_request(300, 50)
        self.preamble_entry.get_buffer().set_text(
            "\n".join(self.machine.preamble)
        )
        self.preamble_entry.get_buffer().connect(
            "changed", self.on_preamble_changed
        )
        preamble_group.add(self.preamble_entry)

        # Postscript
        postscript_group = Adw.PreferencesGroup(title="Postscript")
        gcode_page.add(postscript_group)
        self.postscript_entry = Gtk.TextView()
        self.postscript_entry.set_size_request(300, 50)
        self.postscript_entry.get_buffer().set_text(
            "\n".join(self.machine.postscript)
        )
        self.postscript_entry.get_buffer().connect(
            "changed", self.on_postscript_changed
        )
        postscript_group.add(self.postscript_entry)

        # Air Assist Settings
        air_assist_group = Adw.PreferencesGroup(title="Air Assist")
        gcode_page.add(air_assist_group)

        # Air Assist Enable
        self.air_assist_on_row = Adw.EntryRow()
        gcode = self.machine.air_assist_on or ""
        self.air_assist_on_row.set_title(
            "Air Assist Enable GCode (blank if unsupported)"
        )
        self.air_assist_on_row.set_text(gcode)
        self.air_assist_on_row.connect(
            "changed", self.on_air_assist_on_changed
        )
        air_assist_group.add(self.air_assist_on_row)

        # Air Assist Disable
        self.air_assist_off_row = Adw.EntryRow()
        gcode = self.machine.air_assist_off or ""
        self.air_assist_off_row.set_title(
            "Air Assist Disable GCode (blank if unsupported)"
        )
        self.air_assist_off_row.set_text(gcode)
        self.air_assist_off_row.connect(
            "changed", self.on_air_assist_off_changed
        )
        air_assist_group.add(self.air_assist_off_row)

        # Create the "Laser Heads" page
        laserhead_page = Adw.PreferencesPage(
            title="Laser Heads",
            icon_name="preferences-other-symbolic"
        )
        self.add(laserhead_page)

        # Create the "Camera" page
        camera_page = Adw.PreferencesPage(
            title="Camera",
            icon_name="camera-photo-symbolic"
        )
        self.add(camera_page)

        # List of Cameras
        camera_list_group = Adw.PreferencesGroup(title="Cameras")
        camera_page.add(camera_list_group)
        self.camera_list = Gtk.ListBox()
        self.camera_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.camera_list.set_show_separators(True)
        camera_list_group.add(self.camera_list)

        # Add and Remove buttons for cameras
        camera_button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=5,
            halign=Gtk.Align.END
        )
        add_camera_button = Gtk.Button(icon_name="list-add-symbolic")
        add_camera_button.connect("clicked", self.on_add_camera)
        remove_camera_button = Gtk.Button(icon_name="list-remove-symbolic")
        remove_camera_button.connect("clicked", self.on_remove_camera)
        camera_button_box.append(add_camera_button)
        camera_button_box.append(remove_camera_button)
        camera_list_group.add(camera_button_box)

        # Configuration panel for the selected Camera
        self.camera_properties_widget = CameraProperties(None)
        camera_page.add(self.camera_properties_widget)

        # Connect signals for cameras
        self.camera_list.connect("row-selected", self.on_camera_selected)

        # Populate the list with existing Cameras
        self.populate_camera_list()

        # List of Lasers
        laserhead_list_group = Adw.PreferencesGroup(title="Laser Heads")
        laserhead_page.add(laserhead_list_group)
        self.laserhead_list = Gtk.ListBox()
        self.laserhead_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.laserhead_list.set_show_separators(True)  # Add separators
        laserhead_list_group.add(self.laserhead_list)

        # Add and Remove buttons (right-aligned)
        button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=5,
            halign=Gtk.Align.END
        )
        add_button = Gtk.Button(icon_name="list-add-symbolic")
        add_button.connect("clicked", self.on_add_laserhead)
        remove_button = Gtk.Button(icon_name="list-remove-symbolic")
        remove_button.connect("clicked", self.on_remove_laserhead)
        button_box.append(add_button)
        button_box.append(remove_button)
        laserhead_list_group.add(button_box)

        # Configuration panel for the selected Laser
        self.laserhead_config_group = Adw.PreferencesGroup(
            title="Laser Properties",
            description="Configure the selected laser"
        )
        laserhead_page.add(self.laserhead_config_group)

        max_power_adjustment = Gtk.Adjustment(
            value=0,
            lower=0,
            upper=10000,
            step_increment=1,
            page_increment=10
        )
        self.max_power_row = Adw.SpinRow(
            title="Max Power",
            subtitle="Maximum power value in GCode",
            adjustment=max_power_adjustment
        )
        self.max_power_row.connect("changed", self.on_max_power_changed)
        self.laserhead_config_group.add(self.max_power_row)

        frame_power_adjustment = Gtk.Adjustment(
            value=0,
            lower=0,
            upper=100,
            step_increment=1,
            page_increment=10
        )
        self.frame_power_row = Adw.SpinRow(
            title="Frame Power",
            subtitle="Power value in Gcode to use when framing. 0 to disable",
            adjustment=frame_power_adjustment
        )
        self.frame_power_row.connect("changed", self.on_frame_power_changed)
        self.laserhead_config_group.add(self.frame_power_row)

        spot_size_x_adjustment = Gtk.Adjustment(
            value=0.1,
            lower=0.01,
            upper=0.2,
            step_increment=0.01,
            page_increment=0.05
        )
        self.spot_size_x_row = Adw.SpinRow(
            title="Spot Size X",
            subtitle="Size of the laser spot in the X direction",
            digits=3,
            adjustment=spot_size_x_adjustment
        )
        self.spot_size_x_row.connect("changed", self.on_spot_size_changed)
        self.laserhead_config_group.add(self.spot_size_x_row)

        spot_size_y_adjustment = Gtk.Adjustment(
            value=0.1,
            lower=0.01,
            upper=0.2,
            step_increment=0.01,
            page_increment=0.05
        )
        self.spot_size_y_row = Adw.SpinRow(
            title="Spot Size Y",
            subtitle="Size of the laser spot in the Y direction",
            digits=3,
            adjustment=spot_size_y_adjustment
        )
        self.spot_size_y_row.connect("changed", self.on_spot_size_changed)
        self.laserhead_config_group.add(self.spot_size_y_row)

        # Connect signals
        self.laserhead_list.connect("row-selected", self.on_laserhead_selected)

        # Populate the list with existing Lasers
        self.populate_laserhead_list()

    def populate_laserhead_list(self):
        """Populate the list of Lasers."""
        for head in self.machine.heads:
            row = Adw.ActionRow(title=f"Laser (Max Power: {head.max_power})")
            row.set_margin_top(5)
            row.set_margin_bottom(5)
            self.laserhead_list.append(row)
        row = self.laserhead_list.get_row_at_index(0)
        if row:
            self.laserhead_list.select_row(row)

    def populate_camera_list(self):
        """Populate the list of Cameras."""
        for camera in self.machine.cameras:
            row = Adw.ActionRow(title=f"Camera: {camera.name}")
            row.set_margin_top(5)
            row.set_margin_bottom(5)
            self.camera_list.append(row)
        row = self.camera_list.get_row_at_index(0)
        if row:
            self.camera_list.select_row(row)

    def on_driver_param_changed(self, sender):
        self.machine.set_driver_args(self.driver_group.get_values())

    def on_factory_setup(self, factory, list_item):
        row = Adw.ActionRow()
        list_item.set_child(row)

    def on_factory_bind(self, factory, list_item):
        index = list_item.get_position()
        driver_cls = drivers[index]
        row = list_item.get_child()
        row.set_title(driver_cls.label)
        row.set_subtitle(driver_cls.subtitle)

    def on_combo_row_changed(self, combo_row, _):
        selected_index = combo_row.get_selected()
        driver_cls = drivers[selected_index]

        # This is a workaround due to an Adw.ComboRow bug.
        # Update the ComboRow title to reflect the selected item.
        self.combo_row.set_title(driver_cls.label)
        self.combo_row.set_subtitle(driver_cls.subtitle)

        self.machine.set_driver(driver_cls)
        self.driver_group.create_params(get_params(driver_cls))

    def on_add_laserhead(self, button):
        """Add a new Laser to the machine."""
        new_head = Laser()
        self.machine.add_head(new_head)
        row = Adw.ActionRow(title=f"Laser (Max Power: {new_head.max_power})")
        row.set_margin_top(5)
        row.set_margin_bottom(5)
        self.laserhead_list.append(row)
        self.laserhead_list.select_row(row)

    def on_remove_laserhead(self, button):
        """Remove the selected Laser from the machine."""
        selected_row = self.laserhead_list.get_selected_row()
        if selected_row:
            index = selected_row.get_index()
            head = self.machine.heads[index]
            self.machine.remove_head(head)
            self.laserhead_list.remove(selected_row)

    def on_laserhead_selected(self, listbox, row):
        """Update the configuration panel when a Laser is selected."""
        if row is not None:
            index = row.get_index()
            selected_head = self.machine.heads[index]
            self.max_power_row.set_value(selected_head.max_power)
            self.frame_power_row.set_value(selected_head.frame_power)
            spot_x, spot_y = selected_head.spot_size_mm
            self.spot_size_x_row.set_value(spot_x)
            self.spot_size_y_row.set_value(spot_y)

    def on_add_camera(self, button):
        """Add a new Camera to the machine."""
        dialog = CameraSelectionDialog(self.get_ancestor(Gtk.Window))
        dialog.present()
        dialog.connect("response", self.on_camera_selection_dialog_response)

    def on_camera_selection_dialog_response(self, dialog, response_id):
        if response_id == "select":
            device_id = dialog.selected_device_id
            if device_id:
                # Check if a camera with this device_id already exists
                if any(c.device_id == device_id for c in self.machine.cameras):
                    # Optionally, show a message to the user that camera exists
                    return

                new_camera = Camera(f"Camera {device_id}", device_id)
                self.machine.add_camera(new_camera)
                row = Adw.ActionRow(title=f"Camera: {new_camera.name}")
                row.set_margin_top(5)
                row.set_margin_bottom(5)
                self.camera_list.append(row)
                self.camera_list.select_row(row)
        dialog.destroy()

    def on_remove_camera(self, button):
        """Remove the selected Camera from the machine."""
        selected_row = self.camera_list.get_selected_row()
        if selected_row:
            index = selected_row.get_index()
            camera = self.machine.cameras[index]
            self.machine.remove_camera(camera)
            self.camera_list.remove(selected_row)
            self.camera_properties_widget.set_camera(None)

    def on_camera_selected(self, listbox, row):
        """Update the configuration panel when a Camera is selected."""
        if row is not None:
            index = row.get_index()
            selected_camera = self.machine.cameras[index]
            self.camera_properties_widget.set_camera(selected_camera)
        else:
            self.camera_properties_widget.set_camera(None)

    def _get_selected_laser(self):
        selected_row = self.laserhead_list.get_selected_row()
        if not selected_row:
            return None
        index = selected_row.get_index()
        return self.machine.heads[index]

    def on_max_power_changed(self, spinrow):
        """Update the max power of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        selected_laser.set_max_power(get_spinrow_int(spinrow))
        self.update_laserhead_list()

    def on_frame_power_changed(self, spinrow):
        """Update the max power of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        selected_laser.set_frame_power(get_spinrow_int(spinrow))
        self.update_laserhead_list()

    def on_spot_size_changed(self, spinrow):
        """Update the spot size of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        x = get_spinrow_float(self.spot_size_x_row)
        y = get_spinrow_float(self.spot_size_y_row)
        selected_laser.set_spot_size(x, y)
        self.update_laserhead_list()

    def update_laserhead_list(self):
        """Update the labels in the Laser list."""
        for i, row in enumerate(self.laserhead_list):
            head = self.machine.heads[i]
            row.set_title(f"Laser (Max Power: {head.max_power})")

    def on_preamble_changed(self, buffer):
        """Update the preamble when the text changes."""
        text = buffer.get_text(
            buffer.get_start_iter(),
            buffer.get_end_iter(),
            True
        )
        self.machine.set_preamble(text.splitlines())

    def on_postscript_changed(self, buffer):
        """Update the postscript when the text changes."""
        text = buffer.get_text(
            buffer.get_start_iter(),
            buffer.get_end_iter(),
            True
        )
        self.machine.set_postscript(text.splitlines())

    def on_air_assist_on_changed(self, entry):
        """Update the air assist enable GCode when the value changes."""
        text = entry.get_text().strip()
        self.machine.set_air_assist_on(text if text else None)

    def on_air_assist_off_changed(self, entry):
        """Update the air assist disable GCode when the value changes."""
        text = entry.get_text().strip()
        self.machine.set_air_assist_off(text if text else None)

    def on_home_on_start_changed(self, row, _):
        self.machine.set_home_on_start(row.get_active())

    def on_travel_speed_changed(self, spinrow):
        """Update the max travel speed when the value changes."""
        value = get_spinrow_int(spinrow)
        self.machine.set_max_travel_speed(value)

    def on_cut_speed_changed(self, spinrow):
        """Update the max cut speed when the value changes."""
        value = get_spinrow_int(spinrow)
        self.machine.set_max_cut_speed(value)

    def on_width_changed(self, spinrow):
        """Update the width when the value changes."""
        width = get_spinrow_int(spinrow)
        height = self.machine.dimensions[1]
        self.machine.set_dimensions(width, height)

    def on_height_changed(self, spinrow):
        """Update the height when the value changes."""
        width = self.machine.dimensions[0]
        height = get_spinrow_int(spinrow)
        self.machine.set_dimensions(width, height)
