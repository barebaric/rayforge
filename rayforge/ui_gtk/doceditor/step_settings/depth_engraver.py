from typing import Dict, Any, TYPE_CHECKING, cast
from gi.repository import Gtk, Adw, GObject
from .base import StepComponentSettingsWidget
from ....pipeline.producer.base import OpsProducer
from ....pipeline.producer.depth import DepthEngraver, DepthMode
from ...shared.adwfix import get_spinrow_int, get_spinrow_float
from ....shared.util.glib import DebounceMixin

if TYPE_CHECKING:
    from ....core.step import Step
    from ....doceditor.editor import DocEditor


class DepthEngraverSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the DepthEngraver producer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        producer = cast(DepthEngraver, OpsProducer.from_dict(target_dict))

        super().__init__(
            editor,
            title,
            target_dict=target_dict,
            page=page,
            step=step,
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
        self.min_power_adj = Gtk.Adjustment(
            lower=0,
            upper=100,
            step_increment=1,
            value=producer.min_power * 100,
        )
        self.min_power_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=self.min_power_adj,
            digits=0,
            draw_value=True,
        )
        self.min_power_scale.set_size_request(200, -1)
        self.min_power_row = Adw.ActionRow(
            title=_("Min Power (White)"),
            subtitle=_(
                "Power for lightest areas, as a % of the step's main power"
            ),
        )
        self.min_power_row.add_suffix(self.min_power_scale)
        self.add(self.min_power_row)

        self.max_power_adj = Gtk.Adjustment(
            lower=0,
            upper=100,
            step_increment=1,
            value=producer.max_power * 100,
        )
        self.max_power_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=self.max_power_adj,
            digits=0,
            draw_value=True,
        )
        self.max_power_scale.set_size_request(200, -1)
        self.max_power_row = Adw.ActionRow(
            title=_("Max Power (Black)"),
            subtitle=_(
                "Power for darkest areas, as a % of the step's main power"
            ),
        )
        self.max_power_row.add_suffix(self.max_power_scale)
        self.add(self.max_power_row)

        # --- Multi-Pass Settings ---
        levels_adj = Gtk.Adjustment(
            lower=1,
            upper=255,
            step_increment=1,
            value=producer.num_depth_levels,
        )
        self.levels_row = Adw.SpinRow(
            title=_("Number of Depth Levels"), adjustment=levels_adj
        )
        self.add(self.levels_row)

        z_step_adj = Gtk.Adjustment(
            lower=0, upper=50, step_increment=0.1, value=producer.z_step_down
        )
        self.z_step_row = Adw.SpinRow(
            title=_("Z Step-Down per Level (mm)"),
            adjustment=z_step_adj,
            digits=2,
        )
        self.add(self.z_step_row)

        # Connect signals
        mode_row.connect("notify::selected", self._on_mode_changed)

        self.min_power_handler_id = self.min_power_scale.connect(
            "value-changed", self._on_min_power_scale_changed
        )
        self.max_power_handler_id = self.max_power_scale.connect(
            "value-changed", self._on_max_power_scale_changed
        )

        self.levels_row.connect(
            "changed",
            lambda r: self._debounce(
                self._on_param_changed,
                "num_depth_levels",
                get_spinrow_int(r),
            ),
        )
        self.z_step_row.connect(
            "changed",
            lambda r: self._debounce(
                self._on_param_changed, "z_step_down", get_spinrow_float(r)
            ),
        )

        self._on_mode_changed(mode_row, None)

    def _commit_power_range_change(self):
        """Commits the min/max power to the model via command(s)."""
        min_p = self.min_power_adj.get_value() / 100.0
        max_p = self.max_power_adj.get_value() / 100.0

        params = self.target_dict.setdefault("params", {})
        min_changed = abs(params.get("min_power", 0.0) - min_p) > 1e-6
        max_changed = abs(params.get("max_power", 0.0) - max_p) > 1e-6

        if not min_changed and not max_changed:
            return

        with self.history_manager.transaction(_("Change Power Range")):
            if min_changed:
                self.editor.step.set_step_param(
                    params, "min_power", min_p, _("Change Min Power")
                )
            if max_changed:
                self.editor.step.set_step_param(
                    params, "max_power", max_p, _("Change Max Power")
                )
        self.step.updated.send(self.step)

    def _on_min_power_scale_changed(self, scale: Gtk.Scale):
        new_min_value = self.min_power_adj.get_value()

        # Block the other slider's handler to prevent feedback.
        GObject.signal_handler_block(
            self.max_power_scale, self.max_power_handler_id
        )

        # If the min slider has been dragged past the max slider, push the max
        # slider's value up to match.
        if self.max_power_adj.get_value() < new_min_value:
            self.max_power_adj.set_value(new_min_value)

        # Re-enable the other handler.
        GObject.signal_handler_unblock(
            self.max_power_scale, self.max_power_handler_id
        )

        # Debounce the value that the user is actively changing.
        self._debounce(self._commit_power_range_change)

    def _on_max_power_scale_changed(self, scale: Gtk.Scale):
        new_max_value = self.max_power_adj.get_value()

        # Block the other slider's handler to prevent feedback.
        GObject.signal_handler_block(
            self.min_power_scale, self.min_power_handler_id
        )

        # If the max slider has been dragged past the min slider, push the min
        # slider's value down to match.
        if self.min_power_adj.get_value() > new_max_value:
            self.min_power_adj.set_value(new_max_value)

        # Re-enable the other handler.
        GObject.signal_handler_unblock(
            self.min_power_scale, self.min_power_handler_id
        )

        # Debounce the value that the user is actively changing.
        self._debounce(self._commit_power_range_change)

    def _on_mode_changed(self, row, _):
        selected_idx = row.get_selected()
        selected_mode = list(DepthMode)[selected_idx]
        is_power_mode = selected_mode == DepthMode.POWER_MODULATION

        self.min_power_row.set_visible(is_power_mode)
        self.max_power_row.set_visible(is_power_mode)

        self.levels_row.set_visible(not is_power_mode)
        self.z_step_row.set_visible(not is_power_mode)

        self._on_param_changed("depth_mode", selected_mode.name)

    def _on_param_changed(self, key: str, value: Any):
        target_dict = self.target_dict.setdefault("params", {})
        self.editor.step.set_step_param(
            target_dict=target_dict,
            key=key,
            new_value=value,
            name=_("Change Depth Engraving setting"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
