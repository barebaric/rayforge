from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, Gtk

from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.step_settings.base import (
    StepComponentSettingsWidget,
)
from rayforge.ui_gtk.shared.adwfix import get_spinrow_float

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class WavefrontSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the WavefrontStep."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        page: Adw.PreferencesPage,
        step: Any,
        **kwargs,
    ):
        super().__init__(
            editor,
            title,
            page=page,
            step=step,
            **kwargs,
        )

        laser = self.get_selected_laser()
        default_step_over_mm = laser.spot_size_mm[0] if laser else 0.1
        step_over = (
            step.step_over_mm
            if step.step_over_mm is not None
            else default_step_over_mm
        )
        self._add_spin_row(
            label=_("Step Over"),
            subtitle=_("Lateral step-over between wavefront passes (mm)"),
            value=step_over,
            lower=0.05,
            upper=50.0,
            step_increment=0.1,
            key="step_over_mm",
        )

        self._add_spin_row(
            label=_("Offset"),
            subtitle=_("Extra offset from walls (mm)"),
            value=step.offset_mm,
            lower=0.0,
            upper=20.0,
            step_increment=0.1,
            key="offset_mm",
        )

    def _add_spin_row(
        self,
        label: str,
        subtitle: str,
        value: float,
        lower: float,
        upper: float,
        step_increment: float,
        key: str,
    ):
        adj = Gtk.Adjustment(
            lower=lower,
            upper=upper,
            step_increment=step_increment,
            page_increment=step_increment * 10,
            value=value,
        )
        row = Adw.SpinRow(
            title=label,
            subtitle=subtitle,
            adjustment=adj,
            digits=2,
        )
        self.add(row)
        row.connect(
            "changed",
            lambda r, k=key: self._debounce(
                self._on_param_changed, k, get_spinrow_float(r)
            ),
        )

    def _on_param_changed(self, key: str, new_value: Any):
        self.set_step_property(key, new_value)
