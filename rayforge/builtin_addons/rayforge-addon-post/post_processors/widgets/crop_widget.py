from typing import TYPE_CHECKING

from gi.repository import Adw, Gtk
from gettext import gettext as _

from rayforge.context import get_context
from ..transformers import CropTransformer
from rayforge.ui_gtk.doceditor.step_settings.base import (
    StepComponentSettingsWidget,
)
from rayforge.ui_gtk.shared.adwfix import get_spinrow_float
from rayforge.shared.util.glib import DebounceMixin

if TYPE_CHECKING:
    from rayforge.core.step import Step
    from rayforge.doceditor.editor import DocEditor


class CropTransformerSettingsWidget(
    DebounceMixin, StepComponentSettingsWidget
):
    """UI for configuring the CropTransformer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        transformer: CropTransformer,
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        machine = get_context().machine
        target_dict = step.opsproducer_dict
        if machine and target_dict and "tolerance" not in target_dict:
            target_dict["tolerance"] = machine.arc_tolerance

        super().__init__(
            editor,
            title,
            component=transformer,
            page=page,
            step=step,
            description=transformer.description,
            **kwargs,
        )

        offset_adj = Gtk.Adjustment(
            lower=-100.0, upper=100.0, step_increment=0.1, page_increment=1.0
        )
        offset_row = Adw.SpinRow(
            title=_("Offset"),
            adjustment=offset_adj,
            digits=3,
        )
        offset_row.set_subtitle(
            _("Grow/shrink stock boundary before cropping")
        )
        offset_adj.set_value(transformer.offset)
        self.add(offset_row)
        self.offset_row = offset_row

        offset_row.connect(
            "changed",
            lambda r: self._debounce(self._on_offset_changed, r),
        )

    def _set_step_param(self, key, new_value, name):
        """Helper method to set a step parameter with standard callback."""
        self.editor.step.set_step_param(
            target_dict=self.target_dict,
            key=key,
            new_value=new_value,
            name=name,
            on_change_callback=lambda: self.step.updated.send(self.step),
        )

    def _on_offset_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        self._set_step_param("offset", new_value, _("Change Crop Offset"))
