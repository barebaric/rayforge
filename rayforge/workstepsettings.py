import gi
from blinker import Signal
from rayforge.models import config

gi.require_version('Adw', '1')
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Adw  # noqa: E402


class WorkStepSettingsDialog(Adw.PreferencesDialog):
    def __init__(self, workstep, **kwargs):
        super().__init__(**kwargs)
        self.workstep = workstep

        # Create a preferences page
        page = Adw.PreferencesPage()
        self.add(page)

        # Create a preferences group
        group = Adw.PreferencesGroup(title="Workstep Settings")
        page.add(group)

        # Add a slider for power
        power_row = Adw.ActionRow(title="Power (%)")
        power_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=Gtk.Adjustment(
                value=workstep.power/workstep.laser.max_power*100,
                upper=100,
                step_increment=1,
                page_increment=10
            ),
            digits=0,  # No decimal places
            draw_value=True  # Show the current value
        )
        power_scale.set_size_request(300, -1)
        power_scale.connect('value-changed', self.on_power_changed)
        power_row.add_suffix(power_scale)
        group.add(power_row)

        # Add a spin row for cut speed
        cut_speed_row = Adw.SpinRow(
            title="Cut Speed (mm/min)",
            subtitle=f"Max: {config.machine.max_cut_speed} mm/min",
            adjustment=Gtk.Adjustment(
                value=workstep.cut_speed,
                lower=0,
                upper=config.machine.max_cut_speed,
                step_increment=1,
                page_increment=100
            )
        )
        cut_speed_row.connect('changed', self.on_cut_speed_changed)
        group.add(cut_speed_row)

        # Add a spin row for travel speed
        travel_speed_row = Adw.SpinRow(
            title="Travel Speed (mm/min)",
            subtitle=f"Max: {config.machine.max_travel_speed} mm/min",
            adjustment=Gtk.Adjustment(
                value=workstep.travel_speed,
                lower=0,
                upper=config.machine.max_travel_speed,
                step_increment=1,
                page_increment=100
            )
        )
        travel_speed_row.connect('changed', self.on_travel_speed_changed)
        group.add(travel_speed_row)

        self.changed = Signal()

    def on_power_changed(self, scale):
        self.workstep.power = self.workstep.laser.max_power/100*scale.get_value()
        self.changed.send(self)

    def on_cut_speed_changed(self, spin_row):
        # Workaround: SpinRow seems to have a bug that the value is not always
        # updated if it was edited using the keyboard in the edit field.
        # i.e. get_value() still returns the previous value.
        # So I convert it manually from text if possible.
        try:
            self.workstep.cut_speed = int(spin_row.get_text())
        except ValueError:
            self.workstep.cut_speed = spin_row.get_value()
        self.changed.send(self)

    def on_travel_speed_changed(self, spin_row):
        try:
            self.workstep.travel_speed = int(spin_row.get_text())
        except ValueError:
            self.workstep.travel_speed = spin_row.get_value()
        self.changed.send(self)
