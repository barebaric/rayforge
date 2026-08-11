from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw, GObject

from rayforge.context import get_context
from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.post_processor.groups import (
    ExpanderHost,
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow

from ..transformers import OverscanTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step


class OverscanSettingsGroup(DebounceMixin, TransformerSettingsGroup):
    """UI for configuring the OverscanTransformer."""

    def __init__(
        self,
        title: str,
        transformer: OverscanTransformer,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        **kwargs,
    ):
        super().__init__(title, transformer, page, step=step, **kwargs)

        self._auto = transformer.auto
        self._previous_cut_speed = step.cut_speed if step is not None else None
        if step is not None:
            step.updated.connect(self._on_step_updated)

            machine = get_context().machine
            if machine:
                machine.changed.connect(self._on_machine_changed)

        # Banner shown when the driver applies overscan itself.
        self.native_banner = Adw.Banner(
            title=_(
                "This machine adds overscan automatically; the setting "
                "has no effect."
            )
        )
        super().add(self.native_banner)

        # Auto mode toggle
        self.auto_row = Adw.SwitchRow(
            title=_("Automatic Distance"),
            subtitle=_(
                "Calculate distance based on speed and acceleration with "
                "safety factor"
            ),
        )
        self.auto_row.set_active(transformer.auto)
        self.add(self.auto_row)

        # Distance setting with unit support
        distance_row = LengthSpinRow(
            _("Overscan Distance"),
            _("Manual distance setting"),
            upper=50.0,
            value_in_base=transformer.distance_mm,
        )
        self.add(distance_row)
        self.distance_row = distance_row  # Store reference for later access

        # Connect signals
        self.auto_row.connect("notify::active", self._on_auto_toggled)
        distance_row.value_changed.connect(
            lambda r: self._debounce(self._on_distance_changed, r),
        )

        self.auto_row.connect(
            "notify::active",
            lambda w, _: self._update_sensitivity(),
        )

        self._update_sensitivity()

    def _is_native_overscan(self) -> bool:
        """Whether the active machine's driver applies overscan itself."""
        machine = get_context().machine
        return bool(machine and machine.driver.native_overscan)

    def is_unsupported(self) -> bool:
        """Enabled overscan that the driver handles itself."""
        if self.enable_switch is None or not self.enable_switch.get_active():
            return False
        return self._is_native_overscan()

    def _update_sensitivity(self) -> None:
        """Update the sensitivity of UI elements based on current state."""
        assert self.enable_switch is not None
        enabled = self.enable_switch.get_active()
        auto = self.auto_row.get_active()

        native = self._is_native_overscan()
        self.native_banner.set_revealed(native)

        # Use the stored references to the rows
        self.enable_switch.set_sensitive(not native)
        self.auto_row.set_sensitive(enabled and not native)
        self.distance_row.set_sensitive(enabled and not auto and not native)

    def _on_auto_toggled(
        self, row: Adw.SwitchRow, _pspec: GObject.ParamSpec
    ) -> None:
        self._auto = row.get_active()
        self.param_changed.send(
            self,
            key="auto",
            value=self._auto,
            name=_("Toggle Auto Overscan"),
        )

        # If auto is enabled, recalculate the distance
        if self._auto:
            self._recalculate_distance()

        self._update_sensitivity()

    def _recalculate_distance(self) -> None:
        """Recalculate the overscan distance based on current step settings."""
        machine = get_context().machine
        if not machine or self.step is None:
            return

        # Calculate new distance
        new_distance = OverscanTransformer.calculate_auto_distance(
            self.step.cut_speed, machine.acceleration
        )

        # Update the distance
        self.param_changed.send(
            self,
            key="distance_mm",
            value=new_distance,
            name=_("Auto Calculate Overscan Distance"),
        )

        # Update the UI
        self.distance_row.set_value_in_base_units(new_distance)

    def _on_step_updated(self, step: "Step") -> None:
        """Handle step updates to recalculate overscan distance if needed."""
        if self._auto and step.cut_speed != self._previous_cut_speed:
            self._previous_cut_speed = step.cut_speed
            self._recalculate_distance()

    def _on_machine_changed(self, machine) -> None:
        """
        Handle machine updates (e.g. acceleration) to recalculate overscan.
        """
        if self._is_native_overscan():
            self._update_sensitivity()
            return
        self._update_sensitivity()
        if self._auto:
            self._recalculate_distance()

    def _on_distance_changed(self, spin_row: LengthSpinRow) -> None:
        # Get the value in base units directly from the row
        new_value = self.distance_row.get_value_in_base_units()

        # If auto is currently enabled, disable it when user manually changes
        # the distance (via +/- buttons or typing)
        if self._auto:
            self._auto = False
            self.param_changed.send(
                self,
                key="auto",
                value=False,
                name=_("Disable Auto Overscan"),
            )

        self.param_changed.send(
            self,
            key="distance_mm",
            value=new_value,
            name=_("Change Overscan Distance"),
        )
