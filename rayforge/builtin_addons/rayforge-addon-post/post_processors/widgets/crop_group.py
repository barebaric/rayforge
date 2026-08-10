from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.step_settings.groups import (
    ExpanderHost,
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow

from ..transformers import CropTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step


class CropSettingsGroup(DebounceMixin, TransformerSettingsGroup):
    """UI for configuring the CropTransformer."""

    def __init__(
        self,
        title: str,
        transformer: CropTransformer,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        **kwargs,
    ):
        super().__init__(title, transformer, page, step=step, **kwargs)

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

    def _on_offset_changed(self, row: LengthSpinRow) -> None:
        new_value = row.get_value_in_base_units()
        self.param_changed.send(
            self, key="offset", value=new_value, name=_("Change Crop Offset")
        )
