from gi.repository import Gtk, Adw
from ..driver import drivers, get_driver_cls, get_params
from ..util.adwfix import get_spinrow_int
from .dynamicprefs import DynamicPreferencesGroup


class GeneralPreferencesPage(Adw.PreferencesPage):
    def __init__(self, machine, **kwargs):
        super().__init__(
            title="General",
            icon_name='preferences-system-symbolic',
            **kwargs
        )
        self.machine = machine

        self.driver_group = DynamicPreferencesGroup(title="Driver Settings")
        self.driver_group.data_changed.connect(self.on_driver_param_changed)
        self.add(self.driver_group)

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
        self.add(machine_group)

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
        self.add(dimensions_group)

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
