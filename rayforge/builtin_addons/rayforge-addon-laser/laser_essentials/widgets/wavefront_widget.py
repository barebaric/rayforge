from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, Gtk

from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.step_settings.base import (
    StepComponentSettingsWidget,
)
from rayforge.ui_gtk.shared.adwfix import get_spinrow_float

from ..producers import WavefrontProducer

if TYPE_CHECKING:
    from rayforge.core.step import Step
    from rayforge.doceditor.editor import DocEditor


class WavefrontSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the WavefrontProducer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        producer: WavefrontProducer,
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        super().__init__(
            editor,
            title,
            component=producer,
            page=page,
            step=step,
            **kwargs,
        )

        self.producer = producer

        step_over = producer.step_over_mm or 0.1
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
            value=producer.offset_mm,
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
        params_dict = self.target_dict.setdefault("params", {})
        self.editor.step.set_step_param(
            target_dict=params_dict,
            key=key,
            new_value=new_value,
            name=_("Change Wavefront Setting"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
