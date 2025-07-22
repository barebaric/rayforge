from gi.repository import Gtk, Adw
from blinker import Signal
from ..config import config
from ..util.adwfix import get_spinrow_int


class WorkStepSettingsDialog(Adw.PreferencesDialog):
    def __init__(self, workstep, **kwargs):
        super().__init__(**kwargs)
        self.workstep = workstep
        self.set_title(_("{name} Settings").format(name=workstep.name))

        # Create a preferences page
        page = Adw.PreferencesPage()
        self.add(page)

        # Create a preferences group
        group = Adw.PreferencesGroup()
        page.add(group)

        # Add a spin row for cut speed
        passes_adjustment = Gtk.Adjustment(
            lower=1,
            upper=100,
            step_increment=1,
            page_increment=10
        )
        passes_row = Adw.SpinRow(
            title=_("Number of Passes"),
            subtitle=_("How often to repeat this workstep"),
            adjustment=passes_adjustment
        )
        passes_adjustment.set_value(workstep.passes)
        passes_row.connect('changed', self.on_passes_changed)
        group.add(passes_row)

        # Add a slider for power
        power_row = Adw.ActionRow(title=_("Power (%)"))
        power_adjustment = Gtk.Adjustment(
            upper=100,
            step_increment=1,
            page_increment=10
        )
        power_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=power_adjustment,
            digits=0,  # No decimal places
            draw_value=True  # Show the current value
        )
        power_adjustment.set_value(
            workstep.power / workstep.laser.max_power * 100
        )
        power_scale.set_size_request(300, -1)
        power_scale.connect('value-changed', self.on_power_changed)
        power_row.add_suffix(power_scale)
        group.add(power_row)

        # Add a spin row for cut speed
        cut_speed_adjustment = Gtk.Adjustment(
            lower=0,
            upper=config.machine.max_cut_speed,
            step_increment=1,
            page_increment=100
        )
        cut_speed_row = Adw.SpinRow(
            title=_("Cut Speed (mm/min)"),
            subtitle=_("Max: {max_cut_speed} mm/min").format(
                max_cut_speed=config.machine.max_cut_speed
            ),
            adjustment=cut_speed_adjustment
        )
        cut_speed_adjustment.set_value(workstep.cut_speed)
        cut_speed_row.connect('changed', self.on_cut_speed_changed)
        group.add(cut_speed_row)

        # Add a spin row for travel speed
        travel_speed_adjustment = Gtk.Adjustment(
            lower=0,
            upper=config.machine.max_travel_speed,
            step_increment=1,
            page_increment=100
        )
        travel_speed_row = Adw.SpinRow(
            title=_("Travel Speed (mm/min)"),
            subtitle=_("Max: {max_travel_speed} mm/min").format(
                max_travel_speed=config.machine.max_travel_speed
            ),
            adjustment=travel_speed_adjustment
        )
        travel_speed_adjustment.set_value(workstep.travel_speed)
        travel_speed_row.connect('changed', self.on_travel_speed_changed)
        group.add(travel_speed_row)

        # Add a switch for air assist
        air_assist_row = Adw.SwitchRow()
        air_assist_row.set_title(_("Air Assist"))
        air_assist_row.set_active(workstep.air_assist)
        air_assist_row.connect('notify::active', self.on_air_assist_changed)
        group.add(air_assist_row)

        self.changed = Signal()

    def on_passes_changed(self, spin_row):
        self.workstep.set_passes(get_spinrow_int(spin_row))
        self.changed.send(self)

    def on_power_changed(self, scale):
        max_power = self.workstep.laser.max_power
        self.workstep.set_power(max_power/100*scale.get_value())
        self.changed.send(self)

    def on_cut_speed_changed(self, spin_row):
        self.workstep.set_cut_speed(get_spinrow_int(spin_row))
        self.changed.send(self)

    def on_travel_speed_changed(self, spin_row):
        self.workstep.set_travel_speed(get_spinrow_int(spin_row))
        self.changed.send(self)

    def on_air_assist_changed(self, row, _):
        self.workstep.set_air_assist(row.get_active())
        self.changed.send(self)
