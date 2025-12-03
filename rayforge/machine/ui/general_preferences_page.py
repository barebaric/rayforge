from gi.repository import Gtk, Adw
from ..driver import drivers
from ...shared.ui.adwfix import get_spinrow_int
from ..models.machine import Machine
from ...shared.ui.varsetwidget import VarSetWidget
from ...shared.ui.unit_spin_row import UnitSpinRowHelper


class GeneralPreferencesPage(Adw.PreferencesPage):
    def __init__(self, machine: Machine, **kwargs):
        super().__init__(
            title=_("General"),
            icon_name="preferences-system-symbolic",
            **kwargs,
        )
        self.machine = machine
        self._is_initializing = True
        self._current_driver_name = self.machine.driver_name

        # Error Banner Group
        error_group = Adw.PreferencesGroup()
        self.add(error_group)

        # Configuration Error Banner
        self.error_banner = Adw.Banner()
        self.error_banner.set_use_markup(True)
        self.error_banner.set_revealed(False)
        error_group.add(self.error_banner)
        # Hide the group if the banner is not revealed to avoid extra spacing
        self.error_banner.connect(
            "notify::revealed",
            lambda banner, _: error_group.set_visible(banner.get_revealed()),
        )
        error_group.set_visible(False)

        # Group for Machine Name
        name_group = Adw.PreferencesGroup()
        self.add(name_group)

        # Machine Name
        name_row = Adw.EntryRow(title=_("Name"))
        name_row.set_text(self.machine.name)
        name_row.connect("notify::text", self.on_name_changed)
        name_group.add(name_row)

        self.driver_group = VarSetWidget(title=_("Driver Settings"))
        self.driver_group.data_changed.connect(self.on_driver_param_changed)
        self.add(self.driver_group)

        # Driver selector
        self.driver_store = Gtk.StringList()
        for d in drivers:
            self.driver_store.append(d.label)
        driver_cls = machine.driver.__class__

        self.combo_row = Adw.ComboRow(
            title=_("Select driver"),
            model=self.driver_store,
        )
        self.combo_row.set_use_subtitle(True)
        self.driver_group.add(self.combo_row)

        # Set up a custom factory to display both title and subtitle in the
        # dropdown
        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self.on_factory_setup)
        factory.connect("bind", self.on_factory_bind)
        self.combo_row.set_factory(factory)

        # Perform the initial population of the driver VarSet
        initial_var_set = driver_cls.get_setup_vars()
        initial_var_set.set_values(self.machine.driver_args)
        self.driver_group.populate(initial_var_set)

        # Connect to the machine's changed signal to get updates
        self.machine.changed.connect(self._on_machine_changed)
        self.connect("destroy", self._on_destroy)

        # Connect the signal for the combo row
        self.combo_row.connect("notify::selected", self.on_combo_row_changed)

        # Now, set the initial selection and update its title/subtitle
        if driver_cls:
            selected_index = drivers.index(driver_cls)
            self.combo_row.set_selected(selected_index)
            # Manually set title/subtitle for the initial state
            self.combo_row.set_title(driver_cls.label)
            self.combo_row.set_subtitle(driver_cls.subtitle)
        else:
            self.combo_row.set_title(_("Select driver"))
            self.combo_row.set_subtitle("")

        # Group for Machine Settings
        machine_group = Adw.PreferencesGroup(title=_("Machine Settings"))
        self.add(machine_group)

        home_on_start_row = Adw.SwitchRow()
        home_on_start_row.set_title(_("Home On Start"))
        home_on_start_row.set_subtitle(
            _("Whether Rayforge will send a homing command when it is started")
        )
        home_on_start_row.set_active(machine.home_on_start)
        home_on_start_row.connect(
            "notify::active", self.on_home_on_start_changed
        )
        machine_group.add(home_on_start_row)

        clear_alarm_row = Adw.SwitchRow()
        clear_alarm_row.set_title(_("Clear Alarm On Connect"))
        clear_alarm_row.set_subtitle(
            _(
                "Automatically send an unlock command if "
                "connected in an ALARM state"
            )
        )
        clear_alarm_row.set_active(machine.clear_alarm_on_connect)
        clear_alarm_row.connect(
            "notify::active", self.on_clear_alarm_on_connect_changed
        )
        machine_group.add(clear_alarm_row)

        # Y-Axis direction switch
        y_axis_switch_row = Adw.SwitchRow(
            title=_("Invert Y Axis Direction"),
            subtitle=_(
                "Enable if your machine's origin is top-left"
                " instead of bottom-left"
            ),
        )
        y_axis_switch_row.set_active(self.machine.y_axis_down)
        y_axis_switch_row.connect("notify::active", self.on_y_axis_toggled)
        machine_group.add(y_axis_switch_row)

        # Max Travel Speed
        travel_speed_adjustment = Gtk.Adjustment(
            lower=0,
            upper=60000,  # Increased upper limit for mm/min
            step_increment=10,
            page_increment=100,
        )
        travel_speed_row = Adw.SpinRow(
            title=_("Max Travel Speed"),
            subtitle=_("Maximum rapid movement speed"),
            adjustment=travel_speed_adjustment,
        )
        self.travel_speed_helper = UnitSpinRowHelper(
            spin_row=travel_speed_row,
            quantity="speed",
        )
        self.travel_speed_helper.set_value_in_base_units(
            self.machine.max_travel_speed
        )
        self.travel_speed_helper.changed.connect(self.on_travel_speed_changed)
        self.travel_speed_row = travel_speed_row
        machine_group.add(travel_speed_row)

        # Max Cut Speed
        cut_speed_adjustment = Gtk.Adjustment(
            lower=0,
            upper=60000,  # Increased upper limit for mm/min
            step_increment=10,
            page_increment=100,
        )
        cut_speed_row = Adw.SpinRow(
            title=_("Max Cut Speed"),
            subtitle=_("Maximum cutting speed"),
            adjustment=cut_speed_adjustment,
        )
        self.cut_speed_helper = UnitSpinRowHelper(
            spin_row=cut_speed_row,
            quantity="speed",
        )
        self.cut_speed_helper.set_value_in_base_units(
            self.machine.max_cut_speed
        )
        self.cut_speed_helper.changed.connect(self.on_cut_speed_changed)
        machine_group.add(cut_speed_row)

        # Acceleration
        acceleration_adjustment = Gtk.Adjustment(
            lower=1,
            upper=10000,
            step_increment=10,
            page_increment=100,
        )
        acceleration_row = Adw.SpinRow(
            title=_("Acceleration"),
            subtitle=_(
                "Machine acceleration. Used only for time estimations."
            ),
            adjustment=acceleration_adjustment,
        )
        self.acceleration_helper = UnitSpinRowHelper(
            spin_row=acceleration_row,
            quantity="acceleration",
        )
        self.acceleration_helper.set_value_in_base_units(
            self.machine.acceleration
        )
        self.acceleration_helper.changed.connect(self.on_acceleration_changed)
        machine_group.add(acceleration_row)

        # Dimensions
        dimensions_group = Adw.PreferencesGroup(title=_("Dimensions"))
        self.add(dimensions_group)

        width_adjustment = Gtk.Adjustment(
            lower=50,
            upper=10000,
            step_increment=1,
            page_increment=10,
        )
        self.width_row = Adw.SpinRow(
            title=_("Width"),
            subtitle=_("Width of the machine in mm"),
            adjustment=width_adjustment,
        )
        width_adjustment.set_value(self.machine.dimensions[0])
        self.width_row.connect("changed", self.on_width_changed)
        dimensions_group.add(self.width_row)

        height_adjustment = Gtk.Adjustment(
            lower=50,
            upper=10000,
            step_increment=1,
            page_increment=10,
        )
        self.height_row = Adw.SpinRow(
            title=_("Height"),
            subtitle=_("Height of the machine in mm"),
            adjustment=height_adjustment,
        )
        height_adjustment.set_value(self.machine.dimensions[1])
        self.height_row.connect("changed", self.on_height_changed)
        dimensions_group.add(self.height_row)

        # Initial check for errors
        self._update_error_state()

        # Initialization is complete.
        self._is_initializing = False

        # Update travel speed row based on driver features
        self._update_travel_speed_state()

    def _on_machine_changed(self, sender, **kwargs):
        """
        Handler for the machine's changed signal. This is triggered when
        the driver or its configuration changes, allowing the UI to update.
        """
        self._update_error_state()

        # ONLY repopulate the driver settings if the driver *class* has
        # actually changed. This prevents a full UI rebuild (and focus loss)
        # when just a parameter value is changed.
        if self.machine.driver_name != self._current_driver_name:
            self._current_driver_name = self.machine.driver_name
            driver_cls = self.machine.driver.__class__
            var_set = driver_cls.get_setup_vars()
            var_set.set_values(self.machine.driver_args)
            self.driver_group.populate(var_set)

        # Update travel speed row based on driver features
        self._update_travel_speed_state()

    def _on_destroy(self, *args):
        """Disconnects signals to prevent memory leaks."""
        self.machine.changed.disconnect(self._on_machine_changed)

    def _update_error_state(self):
        """Shows or hides the error banner based on all possible errors."""
        errors = []
        if self.machine.precheck_error:
            errors.append(self.machine.precheck_error)
        if self.machine.driver and self.machine.driver.setup_error:
            errors.append(self.machine.driver.setup_error)

        if errors:
            full_error_msg = " \n".join(errors)
            self.error_banner.set_title(
                _("<b>Configuration required:</b> {error}").format(
                    error=full_error_msg
                )
            )
            self.error_banner.set_revealed(True)
        else:
            self.error_banner.set_revealed(False)

    def on_driver_param_changed(self, sender, **kwargs):
        if self._is_initializing:
            return
        values = self.driver_group.get_values()
        self.machine.set_driver_args(values)

    def on_factory_setup(self, factory, list_item):
        row = Adw.ActionRow()
        list_item.set_child(row)

    def on_factory_bind(self, factory, list_item):
        index = list_item.get_position()
        driver_cls = drivers[index]
        row = list_item.get_child()
        row.set_title(driver_cls.label)
        row.set_subtitle(driver_cls.subtitle)

    def on_combo_row_changed(self, combo_row, _param):
        if self._is_initializing:
            return

        selected_index = combo_row.get_selected()
        if selected_index < 0:
            self.combo_row.set_title(_("Select driver"))
            self.combo_row.set_subtitle("")
            self.driver_group.clear_dynamic_rows()
            return  # No driver selected

        driver_cls = drivers[selected_index]

        self.combo_row.set_title(driver_cls.label)
        self.combo_row.set_subtitle(driver_cls.subtitle)

        # If the user selected a new driver, update the machine model.
        # The `machine.changed` signal will then trigger _on_machine_changed
        # to update the UI, including the driver settings widgets.
        if self.machine.driver_name != driver_cls.__name__:
            self.machine.set_driver(driver_cls, {})

    def on_name_changed(self, entry_row, _):
        """Update the machine name when the text changes."""
        self.machine.set_name(entry_row.get_text())

    def on_home_on_start_changed(self, row, _):
        self.machine.set_home_on_start(row.get_active())

    def on_clear_alarm_on_connect_changed(self, row, _):
        self.machine.set_clear_alarm_on_connect(row.get_active())

    def on_y_axis_toggled(self, row, _):
        self.machine.set_y_axis_down(row.get_active())

    def on_travel_speed_changed(self, helper: UnitSpinRowHelper):
        """Update the max travel speed when the value changes."""
        if self._is_initializing:
            return
        value = helper.get_value_in_base_units()
        self.machine.set_max_travel_speed(int(value))

    def on_cut_speed_changed(self, helper: UnitSpinRowHelper):
        """Update the max cut speed when the value changes."""
        if self._is_initializing:
            return
        value = helper.get_value_in_base_units()
        self.machine.set_max_cut_speed(int(value))

    def on_acceleration_changed(self, helper: UnitSpinRowHelper):
        """Update the acceleration when the value changes."""
        if self._is_initializing:
            return
        value = helper.get_value_in_base_units()
        self.machine.set_acceleration(int(value))

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

    def _update_travel_speed_state(self):
        """Update the travel speed row based on driver features."""
        if self._is_initializing:
            return

        if self.machine.can_g0_with_speed():
            self.travel_speed_row.set_sensitive(True)
            self.travel_speed_row.set_subtitle(
                _("Maximum rapid movement speed")
            )
        else:
            self.travel_speed_row.set_sensitive(False)
            self.travel_speed_row.set_subtitle(
                _("Not supported by the driver")
            )
