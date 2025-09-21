from typing import Dict, Any, TYPE_CHECKING, cast
from gi.repository import Gtk, Adw
from .base import StepComponentSettingsWidget
from ....pipeline.producer.base import OpsProducer
from ....pipeline.producer.depth import DepthEngraver, DepthMode
from ....undo import DictItemCommand
from ....shared.util.adwfix import get_spinrow_int, get_spinrow_float

if TYPE_CHECKING:
    from ....core.step import Step
    from ....undo import HistoryManager


class DepthEngraverSettingsWidget(StepComponentSettingsWidget):
    """UI for configuring the DepthEngraver producer."""

    def __init__(
        self,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        history_manager: "HistoryManager",
        **kwargs,
    ):
        producer = cast(DepthEngraver, OpsProducer.from_dict(target_dict))

        super().__init__(
            target_dict=target_dict,
            page=page,
            step=step,
            history_manager=history_manager,
            **kwargs,
        )

        # Mode selection dropdown
        mode_choices = [m.name.replace("_", " ").title() for m in DepthMode]
        mode_row = Adw.ComboRow(
            title=_("Mode"), model=Gtk.StringList.new(mode_choices)
        )
        mode_row.set_selected(list(DepthMode).index(producer.depth_mode))
        self.add(mode_row)

        # --- Power Modulation Settings ---
        self.power_mode_group = Adw.PreferencesGroup()
        min_power_adj = Gtk.Adjustment(lower=0, upper=100, step_increment=1)
        min_power_row = Adw.SpinRow(
            title=_("Min Power (%)"), adjustment=min_power_adj
        )
        min_power_adj.set_value(producer.min_power)
        self.power_mode_group.add(min_power_row)

        max_power_adj = Gtk.Adjustment(lower=0, upper=100, step_increment=1)
        max_power_row = Adw.SpinRow(
            title=_("Max Power (%)"), adjustment=max_power_adj
        )
        max_power_adj.set_value(producer.max_power)
        self.power_mode_group.add(max_power_row)

        # --- Multi-Pass Settings ---
        self.multipass_mode_group = Adw.PreferencesGroup()
        power_adj = Gtk.Adjustment(lower=0, upper=100, step_increment=1)
        power_row = Adw.SpinRow(title=_("Power (%)"), adjustment=power_adj)
        power_adj.set_value(producer.power)
        self.multipass_mode_group.add(power_row)

        levels_adj = Gtk.Adjustment(lower=1, upper=255, step_increment=1)
        levels_row = Adw.SpinRow(
            title=_("Number of Depth Levels"), adjustment=levels_adj
        )
        levels_adj.set_value(producer.num_depth_levels)
        self.multipass_mode_group.add(levels_row)

        z_step_adj = Gtk.Adjustment(lower=0, upper=50, step_increment=0.1)
        z_step_row = Adw.SpinRow(
            title=_("Z Step-Down per Level (mm)"),
            adjustment=z_step_adj,
            digits=2,
        )
        z_step_adj.set_value(producer.z_step_down)
        self.multipass_mode_group.add(z_step_row)

        # Connect signals
        mode_row.connect("notify::selected", self._on_mode_changed)
        min_power_row.connect(
            "changed",
            lambda r: self._on_param_changed(
                "min_power", get_spinrow_float(r)
            ),
        )
        max_power_row.connect(
            "changed",
            lambda r: self._on_param_changed(
                "max_power", get_spinrow_float(r)
            ),
        )
        power_row.connect(
            "changed",
            lambda r: self._on_param_changed("power", get_spinrow_float(r)),
        )
        levels_row.connect(
            "changed",
            lambda r: self._on_param_changed(
                "num_depth_levels", get_spinrow_int(r)
            ),
        )
        z_step_row.connect(
            "changed",
            lambda r: self._on_param_changed(
                "z_step_down", get_spinrow_float(r)
            ),
        )

        # Initial visibility setup
        self._on_mode_changed(mode_row, None)

    def _on_mode_changed(self, row, _):
        selected_idx = row.get_selected()
        selected_mode = list(DepthMode)[selected_idx]

        # Conditionally add/remove groups from the parent page
        if selected_mode == DepthMode.POWER_MODULATION:
            if self.multipass_mode_group.get_parent():
                self.page.remove(self.multipass_mode_group)
            if not self.power_mode_group.get_parent():
                self.page.add(self.power_mode_group)
        else:  # Multi-Pass
            if self.power_mode_group.get_parent():
                self.page.remove(self.power_mode_group)
            if not self.multipass_mode_group.get_parent():
                self.page.add(self.multipass_mode_group)

        self._on_param_changed("depth_mode", selected_mode.name)

    def _on_param_changed(self, key: str, value: Any):
        target_dict = self.target_dict.setdefault("params", {})
        if value == target_dict.get(key):
            return

        command = DictItemCommand(
            target_dict=target_dict,
            key=key,
            new_value=value,
            name=_("Change Depth Engraving setting"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)
