from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw

from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.step_settings.groups import (
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows.length_spin_row import LengthSpinRow

from ..transformers import CropTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step
    from rayforge.doceditor.editor import DocEditor


class CropSettingsGroup(DebounceMixin, TransformerSettingsGroup):
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
        super().__init__(
            editor,
            title,
            component=transformer,
            page=page,
            step=step,
            description=transformer.description,
            **kwargs,
        )

        self.offset_row = LengthSpinRow(
            _("Offset"),
            _("Grow/shrink stock boundary before cropping"),
            lower=-100.0,
            upper=100.0,
            value_in_base=transformer.offset,
        )
        self.offset_row.value_changed.connect(
            lambda r: self._debounce(self._on_offset_changed, r)
        )
        self.add(self.offset_row)

    def _set_step_param(self, key, new_value, name):
        """Helper method to set a step parameter with standard callback."""
        self.editor.step.set_step_param(
            target_dict=self.target_dict,
            key=key,
            new_value=new_value,
            name=name,
            on_change_callback=lambda: self.step.updated.send(self.step),
        )

    def _on_offset_changed(self, row):
        new_value = row.get_value_in_base_units()
        self._set_step_param("offset", new_value, _("Change Crop Offset"))
