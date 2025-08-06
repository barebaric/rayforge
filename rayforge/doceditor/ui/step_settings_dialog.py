from typing import Tuple, Dict, Any
from gi.repository import Gtk, Adw, GLib  # type: ignore
from blinker import Signal
from ...config import config
from ...undo import HistoryManager, ChangePropertyCommand, DictItemCommand
from ...pipeline.transformer import OpsTransformer, Smooth
from ...shared.util.adwfix import get_spinrow_int
from ...core.doc import Doc
from ...core.step import Step


class StepSettingsDialog(Adw.Window):
    def __init__(self, doc: Doc, step: Step, **kwargs):
        super().__init__(**kwargs)
        self.doc = doc
        self.step = step
        self.history_manager: HistoryManager = doc.history_manager
        self.set_title(_("{name} Settings").format(name=step.name))

        # Used to delay updates from continuous-change widgets like sliders
        # to avoid excessive updates.
        self._debounce_timer = 0
        self._debounced_callback = None
        self._debounced_args: Tuple = ()

        # Safely get machine properties with sensible fallbacks
        if config.machine:
            max_cut_speed = config.machine.max_cut_speed
            max_travel_speed = config.machine.max_travel_speed
        else:
            # Provide sensible defaults if no machine is configured
            max_cut_speed = 3000  # mm/min
            max_travel_speed = 3000  # mm/min

        # Create a vertical box to hold the header bar and the content
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        # Add a header bar for title and window controls (like close)
        header = Adw.HeaderBar()
        main_box.append(header)

        # Set a reasonable default size to avoid being too narrow
        self.set_default_size(600, 750)

        # Destroy window on close to prevent leaks, as a new one is created
        # each time
        self.set_hide_on_close(False)
        self.connect("close-request", self._on_close_request)

        # The main content area should be scrollable
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        scrolled_window.set_vexpand(True)  # Allow the scrolled area to grow
        main_box.append(scrolled_window)

        # Create a preferences page and add it to the scrollable area
        page = Adw.PreferencesPage()
        scrolled_window.set_child(page)

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
            digits=0,
            draw_value=True,
        )
        max_power = (
            step.laser_dict.get("max_power", 1000) if step.laser_dict else 1000
        )
        power_percent = (step.power / max_power * 100) if max_power > 0 else 0
        power_adjustment.set_value(power_percent)
        power_scale.set_size_request(300, -1)
        power_scale.connect(
            "value-changed",
            lambda scale: self._debounce(self.on_power_changed, scale),
        )
        power_row.add_suffix(power_scale)
        general_group.add(power_row)

        # Add a spin row for cut speed
        cut_speed_adjustment = Gtk.Adjustment(
            lower=0,
            upper=max_cut_speed,
            step_increment=1,
            page_increment=100,
        )
        cut_speed_row = Adw.SpinRow(
            title=_("Cut Speed (mm/min)"),
            subtitle=_("Max: {max_cut_speed} mm/min").format(
                max_cut_speed=max_cut_speed
            ),
            adjustment=cut_speed_adjustment,
        )
        cut_speed_adjustment.set_value(step.cut_speed)
        cut_speed_row.connect("changed", self.on_cut_speed_changed)
        general_group.add(cut_speed_row)

        # Add a spin row for travel speed
        travel_speed_adjustment = Gtk.Adjustment(
            lower=0,
            upper=max_travel_speed,
            step_increment=1,
            page_increment=100,
        )
        travel_speed_row = Adw.SpinRow(
            title=_("Travel Speed (mm/min)"),
            subtitle=_("Max: {max_travel_speed} mm/min").format(
                max_travel_speed=max_travel_speed
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
        if self.step.opstransformers_dicts:
            advanced_group = Adw.PreferencesGroup(
                title=_("Path Post-Processing"),
                description=_(
                    "These steps are applied after path generation and"
                    " can improve quality or reduce job time."
                ),
            )
            page.add(advanced_group)

            for transformer_dict in self.step.opstransformers_dicts:
                transformer = OpsTransformer.from_dict(transformer_dict)
                switch_row = Adw.SwitchRow(
                    title=transformer.label, subtitle=transformer.description
                )
                switch_row.set_active(transformer.enabled)
                advanced_group.add(switch_row)
                switch_row.connect(
                    "notify::active",
                    self.on_transformer_toggled,
                    transformer_dict,
                )

                if isinstance(transformer, Smooth):
                    # Smoothness Amount Setting (Slider)
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

                    # Corner Angle Threshold Setting
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
                    # Using a lambda with a default argument `t=transformer`
                    # captures the current value of `transformer` in the loop.
                    smooth_scale.connect(
                        "value-changed",
                        lambda scale, t_dict=transformer_dict: self._debounce(
                            self.on_smoothness_changed, scale, t_dict
                        ),
                    )
                    corner_angle_row.connect(
                        "changed",
                        self.on_corner_angle_changed,
                        transformer_dict,
                    )

        self.changed = Signal()

    def _on_close_request(self, window):
        # Clean up the debounce timer when the window is closed to prevent
        # a GLib warning.
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)
            self._debounce_timer = 0
        return False  # Allow the window to close

    def _debounce(self, callback, *args):
        """
        Schedules a callback to be executed after a short delay, canceling any
        previously scheduled callback. This prevents excessive updates from
        widgets like sliders.
        """
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)

        self._debounced_callback = callback
        self._debounced_args = args
        # Debounce requests by 150ms
        self._debounce_timer = GLib.timeout_add(
            150, self._commit_debounced_change
        )

    def _commit_debounced_change(self):
        """Executes the debounced callback."""
        if self._debounced_callback:
            self._debounced_callback(*self._debounced_args)

        self._debounce_timer = 0
        self._debounced_callback = None
        self._debounced_args = ()
        return GLib.SOURCE_REMOVE

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
        max_power = (
            self.step.laser_dict.get("max_power", 1000)
            if self.step.laser_dict
            else 1000
        )
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

    def on_smoothness_changed(self, scale, transformer_dict: Dict[str, Any]):
        new_value = int(scale.get_value())
        command = DictItemCommand(
            target_dict=transformer_dict,
            key="amount",
            new_value=new_value,
            name=_("Change smoothness"),
            on_change_callback=lambda: self.step.changed.send(self.step),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_corner_angle_changed(
        self, spin_row, transformer_dict: Dict[str, Any]
    ):
        new_value = get_spinrow_int(spin_row)
        command = DictItemCommand(
            target_dict=transformer_dict,
            key="corner_angle_threshold",
            new_value=new_value,
            name=_("Change corner angle"),
            on_change_callback=lambda: self.step.changed.send(self.step),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_transformer_toggled(
        self, row, pspec, transformer_dict: Dict[str, Any]
    ):
        new_value = row.get_active()
        label = transformer_dict.get("label", "Transformer")
        command = DictItemCommand(
            target_dict=transformer_dict,
            key="enabled",
            new_value=new_value,
            name=_("Toggle '{label}'").format(label=label),
            on_change_callback=lambda: self.step.changed.send(self.step),
        )
        self.history_manager.execute(command)
        self.changed.send(self)
