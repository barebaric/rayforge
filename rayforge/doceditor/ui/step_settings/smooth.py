from typing import Dict, Any, TYPE_CHECKING
from gi.repository import Gtk, Adw
from .base import StepComponentSettingsWidget
from ....shared.ui.adwfix import get_spinrow_int
from ....shared.util.glib import DebounceMixin
from ....pipeline.transformer import Smooth
from ....undo import DictItemCommand

if TYPE_CHECKING:
    from ....core.step import Step
    from ....doceditor.editor import DocEditor


class SmoothSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the Smooth transformer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        transformer = Smooth.from_dict(target_dict)

        super().__init__(
            editor,
            title,
            target_dict=target_dict,
            page=page,
            step=step,
            description=transformer.description,
            **kwargs,
        )

        # Main toggle switch
        switch_row = Adw.SwitchRow(title=_("Enable Smoothing"))
        switch_row.set_active(transformer.enabled)
        self.add(switch_row)

        # Smoothness Amount Setting (Slider)
        amount_row = Adw.ActionRow(title=_("Smoothness"))
        amount_adj = Gtk.Adjustment(
            lower=0, upper=100, step_increment=1, page_increment=10
        )
        amount_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=amount_adj,
            digits=0,
            draw_value=True,
        )
        amount_adj.set_value(transformer.amount)
        amount_scale.set_size_request(200, -1)
        amount_row.add_suffix(amount_scale)
        self.add(amount_row)

        # Corner Angle Threshold Setting
        corner_adj = Gtk.Adjustment(
            lower=0, upper=179, step_increment=1, page_increment=10
        )
        corner_row = Adw.SpinRow(
            title=_("Corner Angle Threshold"),
            subtitle=_(
                "Angles sharper than this are kept as corners (degrees)"
            ),
            adjustment=corner_adj,
        )
        corner_adj.set_value(transformer.corner_angle_threshold)
        self.add(corner_row)

        # Set initial sensitivity
        is_enabled = transformer.enabled
        amount_row.set_sensitive(is_enabled)
        corner_row.set_sensitive(is_enabled)

        # Connect signals
        switch_row.connect("notify::active", self._on_enable_toggled)
        switch_row.connect(
            "notify::active",
            self._on_sensitivity_toggled,
            amount_row,
            corner_row,
        )
        amount_scale.connect(
            "value-changed",
            lambda scale: self._debounce(self._on_amount_changed, scale),
        )
        corner_row.connect(
            "changed",
            lambda spin_row: self._debounce(
                self._on_corner_angle_changed, spin_row
            ),
        )

    def _on_enable_toggled(self, row, pspec):
        new_value = row.get_active()
        command = DictItemCommand(
            target_dict=self.target_dict,
            key="enabled",
            new_value=new_value,
            name=_("Toggle Smoothing"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)

    def _on_sensitivity_toggled(self, row, pspec, amount_row, corner_row):
        is_active = row.get_active()
        amount_row.set_sensitive(is_active)
        corner_row.set_sensitive(is_active)

    def _on_amount_changed(self, scale):
        new_value = int(scale.get_value())
        command = DictItemCommand(
            target_dict=self.target_dict,
            key="amount",
            new_value=new_value,
            name=_("Change smoothness"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)

    def _on_corner_angle_changed(self, spin_row):
        new_value = get_spinrow_int(spin_row)
        command = DictItemCommand(
            target_dict=self.target_dict,
            key="corner_angle_threshold",
            new_value=new_value,
            name=_("Change corner angle"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)
