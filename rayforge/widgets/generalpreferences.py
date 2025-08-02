from gi.repository import Gtk, Adw  # type: ignore
from ..driver import drivers, get_driver_cls, get_params
from ..util.adwfix import get_spinrow_int
from .dynamicprefs import DynamicPreferencesGroup


class GeneralPreferencesPage(Adw.PreferencesPage):
    def __init__(self, machine, **kwargs):
        super().__init__(
            title=_("General"),
            icon_name="preferences-system-symbolic",
            **kwargs,
        )
        self.machine = machine

        # Group for Machine Name
        name_group = Adw.PreferencesGroup()
        self.add(name_group)

        # Machine Name
        name_row = Adw.EntryRow(title=_("Name"))
        name_row.set_text(self.machine.name)
        name_row.connect("notify::text", self.on_name_changed)
        name_group.add(name_row)

        self.driver_group = DynamicPreferencesGroup(title=_("Driver Settings"))
        self.driver_group.data_changed.connect(self.on_driver_param_changed)
        self.add(self.driver_group)

        # Driver selector
        self.driver_store = Gtk.StringList()
        for d in drivers:
            self.driver_store.append(d.label)
        driver_cls = get_driver_cls(machine.driver)
        self.combo_row = Adw.ComboRow(
            title=driver_cls.label if driver_cls else _("Select driver"),
            subtitle=driver_cls.subtitle if driver_cls else None,
            model=self.driver_store,
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
        machine_group = Adw.PreferencesGroup(title=_("Machine Settings"))
        self.add(machine_group)

        home_on_start_row = Adw.SwitchRow()
        home_on_start_row.set_title(_("Home On Start"))
        home_on_start_row.set_subtitle(
            _("Whether Rayforce will send a homing command when it is started")
        )
        home_on_start_row.set_active(machine.home_on_start)
        home_on_start_row.connect(
            "notify::active", self.on_home_on_start_changed
        )
        machine_group.add(home_on_start_row)

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
            upper=10000,
            step_increment=1,
            page_increment=10,
        )
        self.travel_speed_row = Adw.SpinRow(
            title=_("Max Travel Speed"),
            subtitle=_("Maximum travel speed in mm/min"),
            adjustment=travel_speed_adjustment,
        )
        travel_speed_adjustment.set_value(self.machine.max_travel_speed)
        self.travel_speed_row.connect("changed", self.on_travel_speed_changed)
        machine_group.add(self.travel_speed_row)

        # Max Cut Speed
        cut_speed_adjustment = Gtk.Adjustment(
            lower=0,
            upper=10000,
            step_increment=1,
            page_increment=10,
        )
        self.cut_speed_row = Adw.SpinRow(
            title=_("Max Cut Speed"),
            subtitle=_("Maximum cutting speed in mm/min"),
            adjustment=cut_speed_adjustment,
        )
        cut_speed_adjustment.set_value(self.machine.max_cut_speed)
        self.cut_speed_row.connect("changed", self.on_cut_speed_changed)
        machine_group.add(self.cut_speed_row)

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

    def on_name_changed(self, entry_row, _):
        """Update the machine name when the text changes."""
        self.machine.set_name(entry_row.get_text())

    def on_home_on_start_changed(self, row, _):
        self.machine.set_home_on_start(row.get_active())

    def on_y_axis_toggled(self, row, _):
        self.machine.set_y_axis_down(row.get_active())

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
