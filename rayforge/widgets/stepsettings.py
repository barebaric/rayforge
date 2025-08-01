from gi.repository import Gtk, Adw  # type: ignore
from blinker import Signal
import math
from ..config import config
from ..undo import HistoryManager
from ..undo.property_cmd import ChangePropertyCommand
from ..opstransformer import Smooth
from ..util.adwfix import get_spinrow_int
from ..models.doc import Doc
from ..models.step import Step


class StepSettingsDialog(Adw.PreferencesDialog):
    def __init__(self, doc: Doc, step: Step, **kwargs):
        super().__init__(**kwargs)
        self.doc = doc
        self.step = step
        self.history_manager: HistoryManager = doc.history_manager
        self.set_title(_("{name} Settings").format(name=step.name))

        # Create a preferences page
        page = Adw.PreferencesPage()
        self.add(page)

        # General Settings group
        general_group = Adw.PreferencesGroup(title=_("General Settings"))
        page.add(general_group)

        # Add a spin row for passes
        passes_adjustment = Gtk.Adjustment(
            lower=1, upper=100, step_increment=1, page_increment=10
        )
        passes_row = Adw.SpinRow(
            title=_("Number of Passes"),
            subtitle=_("How often to repeat this step"),
            adjustment=passes_adjustment,
        )
        passes_adjustment.set_value(step.passes)
        passes_row.connect("changed", self.on_passes_changed)
        general_group.add(passes_row)

        # Add a slider for power
        power_row = Adw.ActionRow(title=_("Power (%)"))
        power_adjustment = Gtk.Adjustment(
            upper=100, step_increment=1, page_increment=10
        )
        power_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=power_adjustment,
            digits=0,  # No decimal places
            draw_value=True,  # Show the current value
        )
        power_adjustment.set_value(
            step.power / step.laser.max_power * 100
        )
        power_scale.set_size_request(300, -1)
        power_scale.connect("value-changed", self.on_power_changed)
        power_row.add_suffix(power_scale)
        general_group.add(power_row)

        # Add a spin row for cut speed
        cut_speed_adjustment = Gtk.Adjustment(
            lower=0,
            upper=config.machine.max_cut_speed,
            step_increment=1,
            page_increment=100,
        )
        cut_speed_row = Adw.SpinRow(
            title=_("Cut Speed (mm/min)"),
            subtitle=_("Max: {max_cut_speed} mm/min").format(
                max_cut_speed=config.machine.max_cut_speed
            ),
            adjustment=cut_speed_adjustment,
        )
        cut_speed_adjustment.set_value(step.cut_speed)
        cut_speed_row.connect("changed", self.on_cut_speed_changed)
        general_group.add(cut_speed_row)

        # Add a spin row for travel speed
        travel_speed_adjustment = Gtk.Adjustment(
            lower=0,
            upper=config.machine.max_travel_speed,
            step_increment=1,
            page_increment=100,
        )
        travel_speed_row = Adw.SpinRow(
            title=_("Travel Speed (mm/min)"),
            subtitle=_("Max: {max_travel_speed} mm/min").format(
                max_travel_speed=config.machine.max_travel_speed
            ),
            adjustment=travel_speed_adjustment,
        )
        travel_speed_adjustment.set_value(step.travel_speed)
        travel_speed_row.connect("changed", self.on_travel_speed_changed)
        general_group.add(travel_speed_row)

        # Add a switch for air assist
        air_assist_row = Adw.SwitchRow()
        air_assist_row.set_title(_("Air Assist"))
        air_assist_row.set_active(step.air_assist)
        air_assist_row.connect("notify::active", self.on_air_assist_changed)
        general_group.add(air_assist_row)

        # Advanced/Optimization Settings
        if step.opstransformers:
            advanced_group = Adw.PreferencesGroup(
                title=_("Path Post-Processing"),
                description=_(
                    "These steps are applied after path generation and"
                    " can improve quality or reduce job time."
                ),
            )
            page.add(advanced_group)

            for transformer in step.opstransformers:
                switch_row = Adw.SwitchRow(
                    title=transformer.label, subtitle=transformer.description
                )
                switch_row.set_active(transformer.enabled)
                advanced_group.add(switch_row)
                switch_row.connect(
                    "notify::active", self.on_transformer_toggled, transformer
                )

                if isinstance(transformer, Smooth):
                    # --- Smoothness Amount Setting (Slider) ---
                    smooth_amount_row = Adw.ActionRow(title=_("Smoothness"))
                    smooth_adj = Gtk.Adjustment(
                        lower=0, upper=100, step_increment=1, page_increment=10
                    )
                    smooth_scale = Gtk.Scale(
                        orientation=Gtk.Orientation.HORIZONTAL,
                        adjustment=smooth_adj,
                        digits=0,
                        draw_value=True,
                    )
                    smooth_adj.set_value(transformer.amount)
                    smooth_scale.set_size_request(200, -1)
                    smooth_amount_row.add_suffix(smooth_scale)
                    advanced_group.add(smooth_amount_row)

                    # --- Corner Angle Threshold Setting ---
                    corner_angle_adj = Gtk.Adjustment(
                        lower=0, upper=179, step_increment=1, page_increment=10
                    )
                    corner_angle_row = Adw.SpinRow(
                        title=_("Corner Angle Threshold"),
                        subtitle=_(
                            "Angles sharper than this are kept as corners"
                            " (degrees)"
                        ),
                        adjustment=corner_angle_adj,
                    )
                    corner_angle_adj.set_value(
                        transformer.corner_angle_threshold
                    )
                    advanced_group.add(corner_angle_row)

                    # Set initial sensitivity
                    is_enabled = transformer.enabled
                    smooth_amount_row.set_sensitive(is_enabled)
                    corner_angle_row.set_sensitive(is_enabled)

                    # Connect signals
                    switch_row.connect(
                        "notify::active",
                        self.on_smooth_switch_sensitivity_toggled,
                        smooth_amount_row,
                        corner_angle_row,
                    )
                    smooth_scale.connect(
                        "value-changed",
                        self.on_smoothness_changed,
                        transformer,
                    )
                    corner_angle_row.connect(
                        "changed", self.on_corner_angle_changed, transformer
                    )

        self.changed = Signal()

    def on_passes_changed(self, spin_row):
        new_value = get_spinrow_int(spin_row)
        if new_value == self.step.passes:
            return
        command = ChangePropertyCommand(
            target=self.step,
            property_name="passes",
            new_value=new_value,
            setter_method_name="set_passes",
            name=_("Change number of passes"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_power_changed(self, scale):
        max_power = self.step.laser.max_power
        new_value = max_power / 100 * scale.get_value()
        command = ChangePropertyCommand(
            target=self.step,
            property_name="power",
            new_value=new_value,
            setter_method_name="set_power",
            name=_("Change laser power"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_cut_speed_changed(self, spin_row):
        new_value = get_spinrow_int(spin_row)
        if new_value == self.step.cut_speed:
            return
        command = ChangePropertyCommand(
            target=self.step,
            property_name="cut_speed",
            new_value=new_value,
            setter_method_name="set_cut_speed",
            name=_("Change cut speed"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_travel_speed_changed(self, spin_row):
        new_value = get_spinrow_int(spin_row)
        if new_value == self.step.travel_speed:
            return
        command = ChangePropertyCommand(
            target=self.step,
            property_name="travel_speed",
            new_value=new_value,
            setter_method_name="set_travel_speed",
            name=_("Change Travel Speed"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_air_assist_changed(self, row, pspec):
        new_value = row.get_active()
        if new_value == self.step.air_assist:
            return
        command = ChangePropertyCommand(
            target=self.step,
            property_name="air_assist",
            new_value=new_value,
            setter_method_name="set_air_assist",
            name=_("Toggle air assist"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_smooth_switch_sensitivity_toggled(
        self, row, pspec, amount_row, angle_row
    ):
        is_active = row.get_active()
        amount_row.set_sensitive(is_active)
        angle_row.set_sensitive(is_active)

    def on_smoothness_changed(self, scale, transformer):
        new_value = int(scale.get_value())
        command = ChangePropertyCommand(
            target=transformer,
            property_name="amount",
            new_value=new_value,
            name=_("Change smoothness"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_corner_angle_changed(self, spin_row, transformer):
        value_deg = get_spinrow_int(spin_row)
        if math.isclose(transformer.corner_angle_threshold, value_deg):
            return
        command = ChangePropertyCommand(
            target=transformer,
            property_name="corner_angle_threshold",
            new_value=value_deg,
            name=_("Change corner angle"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_transformer_toggled(self, row, pspec, transformer):
        new_value = row.get_active()
        if transformer.enabled == new_value:
            return
        command = ChangePropertyCommand(
            target=transformer,
            property_name="enabled",
            new_value=new_value,
            name=_("Toggle '{label}' visibility").format(
                label=transformer.label
            ),
        )
        self.history_manager.execute(command)
        self.changed.send(self)
